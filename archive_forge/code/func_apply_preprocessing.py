import argparse
import gc
import logging
import os
import sys
import traceback
import types
import time
import json
from pyomo.common.deprecation import deprecated
from pyomo.common.log import is_debug_set
from pyomo.common.tempfiles import TempfileManager
from pyomo.common.fileutils import import_file
from pyomo.common.tee import capture_output
from pyomo.common.dependencies import (
from pyomo.common.collections import Bunch
from pyomo.opt import ProblemFormat
from pyomo.opt.base import SolverFactory
from pyomo.opt.parallel import SolverManagerFactory
from pyomo.dataportal import DataPortal
from pyomo.scripting.interface import (
from pyomo.core import Model, TransformationFactory, Suffix, display
def apply_preprocessing(data, parser=None):
    """
    Execute preprocessing files

    Required:
        parser: Command line parser object

    Returned:
        error: This is true if an error has occurred.
    """
    data.local = Bunch()
    if not data.options.runtime.logging == 'quiet':
        sys.stdout.write('[%8.2f] Applying Pyomo preprocessing actions\n' % (time.time() - start_time))
        sys.stdout.flush()
    global filter_excepthook
    if len(data.options.model.filename) == 0:
        parser.print_help()
        data.error = True
        return data
    if not data.options.preprocess is None:
        for config_value in data.options.preprocess:
            preprocess = import_file(config_value, clear_cache=True)
    for ep in ExtensionPoint(IPyomoScriptPreprocess):
        ep.apply(options=data.options)
    for file in [data.options.model.filename] + data.options.data.files.value():
        if not os.path.exists(file):
            raise IOError('File ' + file + ' does not exist!')
    filter_excepthook = True
    tick = time.time()
    data.local.usermodel = import_file(data.options.model.filename, clear_cache=True)
    data.local.time_initial_import = time.time() - tick
    filter_excepthook = False
    usermodel_dir = dir(data.local.usermodel)
    data.local._usermodel_plugins = []
    for key in modelapi:
        if key in usermodel_dir:

            class TMP(Plugin):
                implements(modelapi[key], service=True)

                def __init__(self):
                    self.fn = getattr(data.local.usermodel, key)

                def apply(self, **kwds):
                    return self.fn(**kwds)
            tmp = TMP()
            data.local._usermodel_plugins.append(tmp)
    if 'pyomo_preprocess' in usermodel_dir:
        if data.options.model.object_name in usermodel_dir:
            msg = "Preprocessing function 'pyomo_preprocess' defined in file '%s', but model is already constructed!"
            raise SystemExit(msg % data.options.model.filename)
        getattr(data.local.usermodel, 'pyomo_preprocess')(options=data.options)
    return data
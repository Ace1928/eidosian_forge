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
def apply_postprocessing(data, instance=None, results=None):
    """
    Apply post-processing steps.

    Required:
        instance:   Problem instance.
        results:    Optimization results object.
    """
    if not data.options.runtime.logging == 'quiet':
        sys.stdout.write('[%8.2f] Applying Pyomo postprocessing actions\n' % (time.time() - start_time))
        sys.stdout.flush()
    for config_value in data.options.postprocess:
        postprocess = import_file(config_value, clear_cache=True)
        if 'pyomo_postprocess' in dir(postprocess):
            postprocess.pyomo_postprocess(data.options, instance, results)
    for ep in ExtensionPoint(IPyomoScriptPostprocess):
        ep.apply(options=data.options, instance=instance, results=results)
    if data.options.runtime.profile_memory >= 1 and pympler_available:
        mem_used = pympler.muppy.get_size(pympler.muppy.get_objects())
        if mem_used > data.local.max_memory:
            data.local.max_memory = mem_used
        print('   Total memory = %d bytes upon termination' % mem_used)
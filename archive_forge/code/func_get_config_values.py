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
def get_config_values(filename):
    if filename.endswith('.yml') or filename.endswith('.yaml'):
        if not yaml_available:
            raise ValueError('ERROR: yaml configuration file specified, but pyyaml is not installed!')
        INPUT = open(filename, 'r')
        val = yaml.load(INPUT, **yaml_load_args)
        INPUT.close()
        return val
    elif filename.endswith('.jsn') or filename.endswith('.json'):
        INPUT = open(filename, 'r')
        val = json.load(INPUT)
        INPUT.close()
        return val
    raise IOError("ERROR: Unexpected configuration file '%s'" % filename)
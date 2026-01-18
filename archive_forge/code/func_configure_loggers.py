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
@deprecated('configure_loggers is deprecated. The Pyomo command uses the PyomoCommandLogContext to update the logger configuration', version='5.7.3')
def configure_loggers(options=None, shutdown=False):
    context = PyomoCommandLogContext(options)
    if shutdown:
        context.options.runtime.logging = 'quiet'
        context.fileLogger = configure_loggers.fileLogger
        context.__exit__(None, None, None)
    else:
        context.__enter__()
        configure_loggers.fileLogger = context.fileLogger
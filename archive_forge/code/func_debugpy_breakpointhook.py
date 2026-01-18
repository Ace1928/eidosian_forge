from importlib import import_module
import os
import warnings
from . import check_modules, prefix_matcher, preimport, vendored
import pydevd  # noqa
import debugpy  # noqa
from _pydevd_bundle import pydevd_constants
from _pydevd_bundle import pydevd_defaults
def debugpy_breakpointhook():
    debugpy.breakpoint()
import argparse
import pkgutil
import warnings
import types as pytypes
from numba.core import errors
from numba._version import get_versions
from numba.core.registry import cpu_target
from numba.tests.support import captured_stdout
def check_filter(modname):
    module_components = modname.split('.')
    return any((not filter_fn(module_components) for filter_fn in module_filters))
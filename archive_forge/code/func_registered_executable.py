import glob
import inspect
import logging
import os
import platform
import importlib.util
import sys
from . import envvar
from .dependencies import ctypes
from .deprecation import deprecated, relocated_module_attribute
@deprecated('pyomo.common.registered_executable(name) has been deprecated; use\n    pyomo.common.Executable(name).path() to get the path or\n    pyomo.common.Executable(name).available() to get a bool indicating\n    file availability.  Equivalent results can be obtained by casting\n    Executable(name) to string or bool.', version='5.6.2')
def registered_executable(name):
    ans = Executable(name)
    if ans.path() is None:
        return None
    else:
        return ans
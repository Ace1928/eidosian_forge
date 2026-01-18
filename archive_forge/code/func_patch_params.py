import functools
import inspect
import platform
import sys
import types
from importlib import import_module
from typing import List, TypeVar
import distutils.filelist
def patch_params(mod_name, func_name):
    """
        Prepare the parameters for patch_func to patch indicated function.
        """
    repl_prefix = 'msvc14_'
    repl_name = repl_prefix + func_name.lstrip('_')
    repl = getattr(msvc, repl_name)
    mod = import_module(mod_name)
    if not hasattr(mod, func_name):
        raise ImportError(func_name)
    return (repl, mod, func_name)
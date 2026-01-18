import functools
import inspect
import platform
import sys
import types
from importlib import import_module
from typing import List, TypeVar
import distutils.filelist
def patch_for_msvc_specialized_compiler():
    """
    Patch functions in distutils to use standalone Microsoft Visual C++
    compilers.
    """
    from . import msvc
    if platform.system() != 'Windows':
        return

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
    msvc14 = functools.partial(patch_params, 'distutils._msvccompiler')
    try:
        patch_func(*msvc14('_get_vc_env'))
    except ImportError:
        pass
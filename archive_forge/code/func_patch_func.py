import functools
import inspect
import platform
import sys
import types
from importlib import import_module
from typing import List, TypeVar
import distutils.filelist
def patch_func(replacement, target_mod, func_name):
    """
    Patch func_name in target_mod with replacement

    Important - original must be resolved by name to avoid
    patching an already patched function.
    """
    original = getattr(target_mod, func_name)
    vars(replacement).setdefault('unpatched', original)
    setattr(target_mod, func_name, replacement)
import pkgutil
import types
import importlib
import warnings
from importlib import import_module
import pytest
import scipy
def find_unexpected_members(mod_name):
    members = []
    module = importlib.import_module(mod_name)
    if hasattr(module, '__all__'):
        objnames = module.__all__
    else:
        objnames = dir(module)
    for objname in objnames:
        if not objname.startswith('_'):
            fullobjname = mod_name + '.' + objname
            if isinstance(getattr(module, objname), types.ModuleType):
                if is_unexpected(fullobjname) and fullobjname not in SKIP_LIST_2:
                    members.append(fullobjname)
    return members
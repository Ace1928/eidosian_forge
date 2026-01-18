from collections.abc import Mapping
import inspect
import importlib
import logging
import sys
import warnings
from .deprecation import deprecated, deprecation_warning, in_testing_environment
from .errors import DeferredImportError
def replace_self_in_globals(self, _globals):
    for k, v in _globals.items():
        if v is self:
            _globals[k] = self._available
        elif v.__class__ is DeferredImportModule and v._indicator_flag is self:
            if v._submodule_name is None:
                _globals[k] = self._module
            else:
                _mod_path = v._submodule_name.split('.')[1:]
                _mod = self._module
                for _sub in _mod_path:
                    _mod = getattr(_mod, _sub)
                _globals[k] = _mod
import sys
import traceback
from types import ModuleType
from _pydevd_bundle.pydevd_constants import DebugInfoHolder
import builtins
import_hook_manager = ImportHookManager(__name__ + '.import_hook', builtins.__import__)
def add_module_name(self, module_name, activate_function):
    self._modules_to_patch[module_name] = activate_function
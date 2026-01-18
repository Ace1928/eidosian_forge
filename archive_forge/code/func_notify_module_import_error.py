import importlib  # noqa: F401
import sys
import threading
@synchronized(_import_error_hooks_lock)
def notify_module_import_error(module_name):
    hooks = _import_error_hooks.get(module_name)
    if hooks:
        for hook in hooks:
            hook(module_name)
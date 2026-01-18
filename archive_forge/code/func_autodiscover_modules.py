import copy
import os
import sys
from importlib import import_module
from importlib.util import find_spec as importlib_find
def autodiscover_modules(*args, **kwargs):
    """
    Auto-discover INSTALLED_APPS modules and fail silently when
    not present. This forces an import on them to register any admin bits they
    may want.

    You may provide a register_to keyword parameter as a way to access a
    registry. This register_to object must have a _registry instance variable
    to access it.
    """
    from django.apps import apps
    register_to = kwargs.get('register_to')
    for app_config in apps.get_app_configs():
        for module_to_search in args:
            try:
                if register_to:
                    before_import_registry = copy.copy(register_to._registry)
                import_module('%s.%s' % (app_config.name, module_to_search))
            except Exception:
                if register_to:
                    register_to._registry = before_import_registry
                if module_has_submodule(app_config.module, module_to_search):
                    raise
import asyncio
import fnmatch
import logging
import os
import sys
import types
import warnings
from contextlib import contextmanager
from bokeh.application.handlers import CodeHandler
from ..util import fullpath
from .state import state
@contextmanager
def record_modules(applications=None, handler=None):
    """
    Records modules which are currently imported.
    """
    app_paths = set()
    if hasattr(handler, '_runner'):
        app_paths.add(os.path.dirname(handler._runner.path))
    for app in applications or ():
        if not app._handlers:
            continue
        for handler in app._handlers:
            if isinstance(handler, CodeHandler):
                break
        else:
            continue
        if hasattr(handler, '_runner'):
            app_paths.add(os.path.dirname(handler._runner.path))
    modules = set(sys.modules)
    yield
    for module_name in set(sys.modules).difference(modules):
        if any((module_name.startswith(imodule) for imodule in IGNORED_MODULES)):
            continue
        module = sys.modules[module_name]
        try:
            spec = getattr(module, '__spec__', None)
            if spec is None:
                filepath = getattr(module, '__file__', None)
                if filepath is None:
                    continue
            else:
                filepath = spec.origin
            filepath = fullpath(filepath)
            if filepath is None or in_denylist(filepath):
                continue
            if not os.path.isfile(filepath):
                continue
            parent_path = os.path.dirname(filepath)
            if any((parent_path == app_path or is_subpath(app_path, parent_path) for app_path in app_paths)):
                _local_modules.add(module_name)
            else:
                _modules.add(module_name)
        except Exception:
            continue
import functools
import importlib.util
import pkgutil
import sys
import types
from oslo_log import log as logging
def load_modules(package, ignore_error=False):
    """Dynamically load all modules from a given package."""
    path = package.__path__
    pkg_prefix = package.__name__ + '.'
    for importer, module_name, is_package in pkgutil.walk_packages(path, pkg_prefix):
        if '.tests.' in module_name or module_name.endswith('.setup'):
            continue
        try:
            module = _import_module(importer, module_name, package)
        except ImportError:
            LOG.error('Failed to import module %s', module_name)
            if not ignore_error:
                raise
        else:
            if module is not None:
                yield module
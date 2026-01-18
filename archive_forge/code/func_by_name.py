import sys
import types
from celery._state import current_app
from celery.exceptions import ImproperlyConfigured, reraise
from celery.utils.imports import load_extension_class_names, symbol_by_name
def by_name(backend=None, loader=None, extension_namespace='celery.result_backends'):
    """Get backend class by name/alias."""
    backend = backend or 'disabled'
    loader = loader or current_app.loader
    aliases = dict(BACKEND_ALIASES, **loader.override_backends)
    aliases.update(load_extension_class_names(extension_namespace))
    try:
        cls = symbol_by_name(backend, aliases)
    except ValueError as exc:
        reraise(ImproperlyConfigured, ImproperlyConfigured(UNKNOWN_BACKEND.strip().format(backend, exc)), sys.exc_info()[2])
    if isinstance(cls, types.ModuleType):
        raise ImproperlyConfigured(UNKNOWN_BACKEND.strip().format(backend, 'is a Python module, not a backend class.'))
    return cls
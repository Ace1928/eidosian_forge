import sys
import types
from celery._state import current_app
from celery.exceptions import ImproperlyConfigured, reraise
from celery.utils.imports import load_extension_class_names, symbol_by_name
def by_url(backend=None, loader=None):
    """Get backend class by URL."""
    url = None
    if backend and '://' in backend:
        url = backend
        scheme, _, _ = url.partition('://')
        if '+' in scheme:
            backend, url = url.split('+', 1)
        else:
            backend = scheme
    return (by_name(backend, loader), url)
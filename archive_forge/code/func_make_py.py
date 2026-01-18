import os
import sys
import importlib.util as imputil
import mimetypes
from paste import request
from paste import fileapp
from paste.util import import_string
from paste import httpexceptions
from .httpheaders import ETAG
from paste.util import converters
def make_py(parser, environ, filename):
    module = load_module(environ, filename)
    if not module:
        return None
    if hasattr(module, 'application') and module.application:
        return getattr(module.application, 'wsgi_application', module.application)
    base_name = module.__name__.split('.')[-1]
    if hasattr(module, base_name):
        obj = getattr(module, base_name)
        if hasattr(obj, 'wsgi_application'):
            return obj.wsgi_application
        else:
            return getattr(module, base_name)()
    environ['wsgi.errors'].write('Cound not find application or %s in %s\n' % (base_name, module))
    return None
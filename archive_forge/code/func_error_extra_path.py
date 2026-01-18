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
def error_extra_path(self, environ, start_response):
    exc = httpexceptions.HTTPNotFound('The trailing path %r is not allowed' % environ['PATH_INFO'])
    return exc.wsgi_application(environ, start_response)
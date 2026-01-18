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
def find_init_module(self, environ):
    filename = os.path.join(self.directory, '__init__.py')
    if not os.path.exists(filename):
        return None
    return load_module(environ, filename)
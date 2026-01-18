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
def make_static(global_conf, document_root, cache_max_age=None):
    """
    Return a WSGI application that serves a directory (configured
    with document_root)

    cache_max_age - integer specifies CACHE_CONTROL max_age in seconds
    """
    if cache_max_age is not None:
        cache_max_age = int(cache_max_age)
    return StaticURLParser(document_root, cache_max_age=cache_max_age)
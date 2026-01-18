from configparser import ConfigParser
import os
import re
import sys
from urllib.parse import unquote
from paste.deploy.util import fix_call, importlib_metadata, lookup_object
def loadcontext(object_type, uri, name=None, relative_to=None, global_conf=None):
    if '#' in uri:
        if name is None:
            uri, name = uri.split('#', 1)
        else:
            uri = uri.split('#', 1)[0]
    if name is None:
        name = 'main'
    if ':' not in uri:
        raise LookupError('URI has no scheme: %r' % uri)
    scheme, path = uri.split(':', 1)
    scheme = scheme.lower()
    if scheme not in _loaders:
        raise LookupError('URI scheme not known: {!r} (from {})'.format(scheme, ', '.join(_loaders.keys())))
    return _loaders[scheme](object_type, uri, path, name=name, relative_to=relative_to, global_conf=global_conf)
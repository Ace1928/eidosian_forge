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
def make_pkg_resources(global_conf, egg, resource_name=''):
    """
    A static file parser that loads data from an egg using
    ``pkg_resources``.  Takes a configuration value ``egg``, which is
    an egg spec, and a base ``resource_name`` (default empty string)
    which is the path in the egg that this starts at.
    """
    if pkg_resources is None:
        raise NotImplementedError('This function requires pkg_resources.')
    return PkgResourcesParser(egg, resource_name)
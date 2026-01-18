import abc
import sys
import stat as st
from _collections_abc import _check_methods
from os.path import (curdir, pardir, sep, pathsep, defpath, extsep, altsep,
from _collections_abc import MutableMapping, Mapping
def getenv(key, default=None):
    """Get an environment variable, return None if it doesn't exist.
    The optional second argument can specify an alternate default.
    key, default and the result are str."""
    return environ.get(key, default)
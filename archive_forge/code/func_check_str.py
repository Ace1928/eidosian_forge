import abc
import sys
import stat as st
from _collections_abc import _check_methods
from os.path import (curdir, pardir, sep, pathsep, defpath, extsep, altsep,
from _collections_abc import MutableMapping, Mapping
def check_str(value):
    if not isinstance(value, str):
        raise TypeError('str expected, not %s' % type(value).__name__)
    return value
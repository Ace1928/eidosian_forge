from __future__ import absolute_import
import platform
from ctypes import (
from ctypes.util import find_library
from ...packages.six import raise_from
def load_cdll(name, macos10_16_path):
    """Loads a CDLL by name, falling back to known path on 10.16+"""
    try:
        if version_info >= (10, 16):
            path = macos10_16_path
        else:
            path = find_library(name)
        if not path:
            raise OSError
        return CDLL(path, use_errno=True)
    except OSError:
        raise_from(ImportError('The library %s failed to load' % name), None)
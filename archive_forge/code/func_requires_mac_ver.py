from __future__ import (absolute_import, division,
from future import utils
from future.builtins import str, range, open, int, map, list
import contextlib
import errno
import functools
import gc
import socket
import sys
import os
import platform
import shutil
import warnings
import unittest
import importlib
import re
import subprocess
import time
import fnmatch
import logging.handlers
import struct
import tempfile
def requires_mac_ver(*min_version):
    """Decorator raising SkipTest if the OS is Mac OS X and the OS X
    version if less than min_version.

    For example, @requires_mac_ver(10, 5) raises SkipTest if the OS X version
    is lesser than 10.5.
    """

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kw):
            if sys.platform == 'darwin':
                version_txt = platform.mac_ver()[0]
                try:
                    version = tuple(map(int, version_txt.split('.')))
                except ValueError:
                    pass
                else:
                    if version < min_version:
                        min_version_txt = '.'.join(map(str, min_version))
                        raise unittest.SkipTest('Mac OS X %s or higher required, not %s' % (min_version_txt, version_txt))
            return func(*args, **kw)
        wrapper.min_version = min_version
        return wrapper
    return decorator
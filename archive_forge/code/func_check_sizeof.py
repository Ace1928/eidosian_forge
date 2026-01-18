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
def check_sizeof(test, o, size):
    result = sys.getsizeof(o)
    if type(o) == type and o.__flags__ & _TPFLAGS_HEAPTYPE or (type(o) != type and type(o).__flags__ & _TPFLAGS_HAVE_GC):
        size += _testcapi.SIZEOF_PYGC_HEAD
    msg = 'wrong size for %s: got %d, expected %d' % (type(o), result, size)
    test.assertEqual(result, size, msg)
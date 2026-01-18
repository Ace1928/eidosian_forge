import importlib.util
import os
import re
import string
import subprocess
import sys
import sysconfig
import functools
from .errors import DistutilsPlatformError, DistutilsByteCompileError
from ._modified import newer
from .spawn import spawn
from ._log import log
from distutils.util import byte_compile
def split_version(s):
    """Convert a dot-separated string into a list of numbers for comparisons"""
    return [int(n) for n in s.split('.')]
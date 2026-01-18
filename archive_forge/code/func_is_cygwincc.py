import os
import re
import sys
import copy
import shlex
import warnings
from subprocess import check_output
from .unixccompiler import UnixCCompiler
from .file_util import write_file
from .errors import (
from .version import LooseVersion, suppress_known_deprecation
from ._collections import RangeMap
def is_cygwincc(cc):
    """Try to determine if the compiler that would be used is from cygwin."""
    out_string = check_output(shlex.split(cc) + ['-dumpmachine'])
    return out_string.strip().endswith(b'cygwin')
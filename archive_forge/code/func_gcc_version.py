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
@property
def gcc_version(self):
    warnings.warn('gcc_version attribute of CygwinCCompiler is deprecated. Instead of returning actual gcc version a fixed value 11.2.0 is returned.', DeprecationWarning, stacklevel=2)
    with suppress_known_deprecation():
        return LooseVersion('11.2.0')
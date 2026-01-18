import os
import sys
import copy
from subprocess import Popen, PIPE, check_output
import re
from distutils.unixccompiler import UnixCCompiler
from distutils.file_util import write_file
from distutils.errors import (DistutilsExecError, CCompilerError,
from distutils.version import LooseVersion
from distutils.spawn import find_executable
def is_cygwingcc():
    """Try to determine if the gcc that would be used is from cygwin."""
    out_string = check_output(['gcc', '-dumpmachine'])
    return out_string.strip().endswith(b'cygwin')
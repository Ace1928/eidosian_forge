import os
import sys
import subprocess
import re
import textwrap
import numpy.distutils.ccompiler  # noqa: F401
from numpy.distutils import log
import distutils.cygwinccompiler
from distutils.unixccompiler import UnixCCompiler
from distutils.msvccompiler import get_build_version as get_build_msvc_version
from distutils.errors import UnknownFileError
from numpy.distutils.misc_util import (msvc_runtime_library,
def get_msvcr_replacement():
    """Replacement for outdated version of get_msvcr from cygwinccompiler"""
    msvcr = msvc_runtime_library()
    return [] if msvcr is None else [msvcr]
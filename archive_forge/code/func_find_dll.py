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
def find_dll(dll_name):
    arch = {'AMD64': 'amd64', 'Intel': 'x86'}[get_build_architecture()]

    def _find_dll_in_winsxs(dll_name):
        winsxs_path = os.path.join(os.environ.get('WINDIR', 'C:\\WINDOWS'), 'winsxs')
        if not os.path.exists(winsxs_path):
            return None
        for root, dirs, files in os.walk(winsxs_path):
            if dll_name in files and arch in root:
                return os.path.join(root, dll_name)
        return None

    def _find_dll_in_path(dll_name):
        for path in [sys.prefix] + os.environ['PATH'].split(';'):
            filepath = os.path.join(path, dll_name)
            if os.path.exists(filepath):
                return os.path.abspath(filepath)
    return _find_dll_in_winsxs(dll_name) or _find_dll_in_path(dll_name)
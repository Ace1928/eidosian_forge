import os
import signal
import subprocess
import sys
import textwrap
import warnings
from distutils.command.config import config as old_config
from distutils.command.config import LANG_EXT
from distutils import log
from distutils.file_util import copy_file
from distutils.ccompiler import CompileError, LinkError
import distutils
from numpy.distutils.exec_command import filepath_from_subprocess_output
from numpy.distutils.mingw32ccompiler import generate_manifest
from numpy.distutils.command.autodist import (check_gcc_function_attribute,
def check_type_size(self, type_name, headers=None, include_dirs=None, library_dirs=None, expected=None):
    """Check size of a given type."""
    self._check_compiler()
    body = textwrap.dedent('\n            typedef %(type)s npy_check_sizeof_type;\n            int main (void)\n            {\n                static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) >= 0)];\n                test_array [0] = 0\n\n                ;\n                return 0;\n            }\n            ')
    self._compile(body % {'type': type_name}, headers, include_dirs, 'c')
    self._clean()
    if expected:
        body = textwrap.dedent('\n                typedef %(type)s npy_check_sizeof_type;\n                int main (void)\n                {\n                    static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) == %(size)s)];\n                    test_array [0] = 0\n\n                    ;\n                    return 0;\n                }\n                ')
        for size in expected:
            try:
                self._compile(body % {'type': type_name, 'size': size}, headers, include_dirs, 'c')
                self._clean()
                return size
            except CompileError:
                pass
    body = textwrap.dedent('\n            typedef %(type)s npy_check_sizeof_type;\n            int main (void)\n            {\n                static int test_array [1 - 2 * !(((long) (sizeof (npy_check_sizeof_type))) <= %(size)s)];\n                test_array [0] = 0\n\n                ;\n                return 0;\n            }\n            ')
    low = 0
    mid = 0
    while True:
        try:
            self._compile(body % {'type': type_name, 'size': mid}, headers, include_dirs, 'c')
            self._clean()
            break
        except CompileError:
            low = mid + 1
            mid = 2 * mid + 1
    high = mid
    while low != high:
        mid = (high - low) // 2 + low
        try:
            self._compile(body % {'type': type_name, 'size': mid}, headers, include_dirs, 'c')
            self._clean()
            high = mid
        except CompileError:
            low = mid + 1
    return low
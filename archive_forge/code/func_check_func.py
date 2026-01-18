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
def check_func(self, func, headers=None, include_dirs=None, libraries=None, library_dirs=None, decl=False, call=False, call_args=None):
    self._check_compiler()
    body = []
    if decl:
        if type(decl) == str:
            body.append(decl)
        else:
            body.append('int %s (void);' % func)
    body.append('#ifdef _MSC_VER')
    body.append('#pragma function(%s)' % func)
    body.append('#endif')
    body.append('int main (void) {')
    if call:
        if call_args is None:
            call_args = ''
        body.append('  %s(%s);' % (func, call_args))
    else:
        body.append('  %s;' % func)
    body.append('  return 0;')
    body.append('}')
    body = '\n'.join(body) + '\n'
    return self.try_link(body, headers, include_dirs, libraries, library_dirs)
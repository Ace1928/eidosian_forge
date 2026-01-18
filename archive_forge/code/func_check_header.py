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
def check_header(self, header, include_dirs=None, library_dirs=None, lang='c'):
    self._check_compiler()
    return self.try_compile('/* we need a dummy line to make distutils happy */', [header], include_dirs)
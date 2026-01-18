import sys
import os
import re
import copy
import warnings
import subprocess
import textwrap
from glob import glob
from functools import reduce
from configparser import NoOptionError
from configparser import RawConfigParser as ConfigParser
from distutils.errors import DistutilsError
from distutils.dist import Distribution
import sysconfig
from numpy.distutils import log
from distutils.util import get_platform
from numpy.distutils.exec_command import (
from numpy.distutils.misc_util import (is_sequence, is_string,
from numpy.distutils.command.config import config as cmd_config
from numpy.distutils import customized_ccompiler as _customized_ccompiler
from numpy.distutils import _shell_utils
import distutils.ccompiler
import tempfile
import shutil
import platform
def check_symbols(self, info):
    res = False
    c = customized_ccompiler()
    tmpdir = tempfile.mkdtemp()
    prototypes = '\n'.join(('void %s%s%s();' % (self.symbol_prefix, symbol_name, self.symbol_suffix) for symbol_name in self._require_symbols))
    calls = '\n'.join(('%s%s%s();' % (self.symbol_prefix, symbol_name, self.symbol_suffix) for symbol_name in self._require_symbols))
    s = textwrap.dedent('            %(prototypes)s\n            int main(int argc, const char *argv[])\n            {\n                %(calls)s\n                return 0;\n            }') % dict(prototypes=prototypes, calls=calls)
    src = os.path.join(tmpdir, 'source.c')
    out = os.path.join(tmpdir, 'a.out')
    try:
        extra_args = info['extra_link_args']
    except Exception:
        extra_args = []
    try:
        with open(src, 'w') as f:
            f.write(s)
        obj = c.compile([src], output_dir=tmpdir)
        try:
            c.link_executable(obj, out, libraries=info['libraries'], library_dirs=info['library_dirs'], extra_postargs=extra_args)
            res = True
        except distutils.ccompiler.LinkError:
            res = False
    finally:
        shutil.rmtree(tmpdir)
    return res
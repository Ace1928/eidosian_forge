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
def check_msvc_gfortran_libs(self, library_dirs, libraries):
    library_paths = []
    for library in libraries:
        for library_dir in library_dirs:
            fullpath = os.path.join(library_dir, library + '.a')
            if os.path.isfile(fullpath):
                library_paths.append(fullpath)
                break
        else:
            return None
    basename = self.__class__.__name__
    tmpdir = os.path.join(os.getcwd(), 'build', basename)
    if not os.path.isdir(tmpdir):
        os.makedirs(tmpdir)
    info = {'library_dirs': [tmpdir], 'libraries': [basename], 'language': 'f77'}
    fake_lib_file = os.path.join(tmpdir, basename + '.fobjects')
    fake_clib_file = os.path.join(tmpdir, basename + '.cobjects')
    with open(fake_lib_file, 'w') as f:
        f.write('\n'.join(library_paths))
    with open(fake_clib_file, 'w') as f:
        pass
    return info
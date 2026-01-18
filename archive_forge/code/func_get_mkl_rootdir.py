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
def get_mkl_rootdir(self):
    mklroot = os.environ.get('MKLROOT', None)
    if mklroot is not None:
        return mklroot
    paths = os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep)
    ld_so_conf = '/etc/ld.so.conf'
    if os.path.isfile(ld_so_conf):
        with open(ld_so_conf) as f:
            for d in f:
                d = d.strip()
                if d:
                    paths.append(d)
    intel_mkl_dirs = []
    for path in paths:
        path_atoms = path.split(os.sep)
        for m in path_atoms:
            if m.startswith('mkl'):
                d = os.sep.join(path_atoms[:path_atoms.index(m) + 2])
                intel_mkl_dirs.append(d)
                break
    for d in paths:
        dirs = glob(os.path.join(d, 'mkl', '*'))
        dirs += glob(os.path.join(d, 'mkl*'))
        for sub_dir in dirs:
            if os.path.isdir(os.path.join(sub_dir, 'lib')):
                return sub_dir
    return None
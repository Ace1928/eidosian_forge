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
def add_system_root(library_root):
    """Add a package manager root to the include directories"""
    global default_lib_dirs
    global default_include_dirs
    library_root = os.path.normpath(library_root)
    default_lib_dirs.extend((os.path.join(library_root, d) for d in _lib_dirs))
    default_include_dirs.extend((os.path.join(library_root, d) for d in _include_dirs))
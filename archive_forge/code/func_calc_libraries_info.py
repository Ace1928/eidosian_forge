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
def calc_libraries_info(self):
    libs = self.get_libraries()
    dirs = self.get_lib_dirs()
    r_dirs = self.get_runtime_lib_dirs()
    r_dirs.extend(self.get_runtime_lib_dirs(key='rpath'))
    info = {}
    for lib in libs:
        i = self.check_libs(dirs, [lib])
        if i is not None:
            dict_append(info, **i)
        else:
            log.info('Library %s was not found. Ignoring' % lib)
        if r_dirs:
            i = self.check_libs(r_dirs, [lib])
            if i is not None:
                del i['libraries']
                i['runtime_library_dirs'] = i.pop('library_dirs')
                dict_append(info, **i)
            else:
                log.info('Runtime library %s was not found. Ignoring' % lib)
    return info
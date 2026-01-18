import os
import re
import sys
import shlex
import copy
from distutils.command import build_ext
from distutils.dep_util import newer_group, newer
from distutils.util import get_platform
from distutils.errors import DistutilsError, DistutilsSetupError
from numpy.distutils import log
from numpy.distutils.misc_util import (
from numpy.distutils.from_template import process_file as process_f_file
from numpy.distutils.conv_template import process_file as process_c_file
def build_sources(self):
    if self.inplace:
        self.get_package_dir = self.get_finalized_command('build_py').get_package_dir
    self.build_py_modules_sources()
    for libname_info in self.libraries:
        self.build_library_sources(*libname_info)
    if self.extensions:
        self.check_extensions_list(self.extensions)
        for ext in self.extensions:
            self.build_extension_sources(ext)
    self.build_data_files_sources()
    self.build_npy_pkg_config()
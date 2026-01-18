import os
import importlib.util
import sys
import glob
from distutils.core import Command
from distutils.errors import *
from distutils.util import convert_path, Mixin2to3
from distutils import log
def check_package(self, package, package_dir):
    if package_dir != '':
        if not os.path.exists(package_dir):
            raise DistutilsFileError("package directory '%s' does not exist" % package_dir)
        if not os.path.isdir(package_dir):
            raise DistutilsFileError("supposed package directory '%s' exists, but is not a directory" % package_dir)
    if package:
        init_py = os.path.join(package_dir, '__init__.py')
        if os.path.isfile(init_py):
            return init_py
        else:
            log.warn("package init file '%s' not found " + '(or not a regular file)', init_py)
    return None
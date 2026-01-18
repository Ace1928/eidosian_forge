import os
import importlib.util
import sys
import glob
from distutils.core import Command
from distutils.errors import *
from distutils.util import convert_path, Mixin2to3
from distutils import log
def build_package_data(self):
    """Copy data files into build directory"""
    lastdir = None
    for package, src_dir, build_dir, filenames in self.data_files:
        for filename in filenames:
            target = os.path.join(build_dir, filename)
            self.mkpath(os.path.dirname(target))
            self.copy_file(os.path.join(src_dir, filename), target, preserve_mode=False)
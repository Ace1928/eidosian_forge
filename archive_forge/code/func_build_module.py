import os
import importlib.util
import sys
import glob
from distutils.core import Command
from distutils.errors import *
from distutils.util import convert_path, Mixin2to3
from distutils import log
def build_module(self, module, module_file, package):
    res = build_py.build_module(self, module, module_file, package)
    if res[1]:
        self.updated_files.append(res[0])
    return res
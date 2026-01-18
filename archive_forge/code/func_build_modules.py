import os
import importlib.util
import sys
import glob
from distutils.core import Command
from distutils.errors import *
from distutils.util import convert_path, Mixin2to3
from distutils import log
def build_modules(self):
    modules = self.find_modules()
    for package, module, module_file in modules:
        self.build_module(module, module_file, package)
import os
import importlib.util
import sys
import glob
from distutils.core import Command
from distutils.errors import *
from distutils.util import convert_path, Mixin2to3
from distutils import log
def build_packages(self):
    for package in self.packages:
        package_dir = self.get_package_dir(package)
        modules = self.find_package_modules(package, package_dir)
        for package_, module, module_file in modules:
            assert package == package_
            self.build_module(module, module_file, package)
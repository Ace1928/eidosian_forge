import sys, os
from distutils.core import Command
from distutils.errors import DistutilsOptionError
from distutils.util import get_platform
def has_pure_modules(self):
    return self.distribution.has_pure_modules()
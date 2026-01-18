import sys, os
from distutils.core import Command
from distutils.errors import DistutilsOptionError
from distutils.util import get_platform
def has_scripts(self):
    return self.distribution.has_scripts()
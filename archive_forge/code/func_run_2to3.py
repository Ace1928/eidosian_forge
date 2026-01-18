import os
import re
import importlib.util
import string
import sys
import distutils
from distutils.errors import DistutilsPlatformError
from distutils.dep_util import newer
from distutils.spawn import spawn
from distutils import log
from distutils.errors import DistutilsByteCompileError
from distutils.util import byte_compile
def run_2to3(self, files):
    return run_2to3(files, self.fixer_names, self.options, self.explicit)
import sys
import os
from distutils.errors import DistutilsPlatformError
from distutils.dep_util import newer, newer_group
from distutils import log
from distutils.command import build_ext as _build_ext
from distutils import sysconfig
import inspect
import warnings
def disable_optimization(self):
    """disable optimization for the C or C++ compiler"""
    badoptions = ('-O1', '-O2', '-O3')
    for flag, option in zip(self.flags, self.state):
        if option is not None:
            L = [opt for opt in option.split() if opt not in badoptions]
            self.config_vars[flag] = ' '.join(L)
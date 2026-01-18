from __future__ import division
import sys
import importlib
import logging
import functools
import pkgutil
import io
import numpy as np
from scipy import sparse
import scipy.io
def import_modules(names, src, dst):
    """Import modules in package."""
    for name in names:
        module = importlib.import_module(src + '.' + name)
        setattr(sys.modules[dst], name, module)
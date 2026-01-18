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
def import_functions(names, src, dst):
    """Import functions in package from their implementation modules."""
    for name in names:
        module = importlib.import_module('pygsp.' + src)
        setattr(sys.modules['pygsp.' + dst], name, getattr(module, name))
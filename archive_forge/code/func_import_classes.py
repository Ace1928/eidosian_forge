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
def import_classes(names, src, dst):
    """Import classes in package from their implementation modules."""
    for name in names:
        module = importlib.import_module('pygsp.' + src + '.' + name.lower())
        setattr(sys.modules['pygsp.' + dst], name, getattr(module, name))
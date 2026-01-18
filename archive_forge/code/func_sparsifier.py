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
def sparsifier(func):

    def inner(*args, **kwargs):
        return sparse.lil_matrix(func(*args, **kwargs))
    return inner
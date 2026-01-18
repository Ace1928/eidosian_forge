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
def filterbank_handler(func):

    @functools.wraps(func)
    def inner(f, *args, **kwargs):
        if 'i' in kwargs:
            return func(f, *args, **kwargs)
        elif f.Nf <= 1:
            return func(f, *args, **kwargs)
        else:
            output = []
            for i in range(f.Nf):
                output.append(func(f, *args, i=i, **kwargs))
            return output
    return inner
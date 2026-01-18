from collections import namedtuple
import contextlib
from functools import cache, wraps
import inspect
from inspect import Signature, Parameter
import logging
from numbers import Number, Real
import re
import warnings
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .colors import BoundaryNorm
from .cm import ScalarMappable
from .path import Path
from .transforms import (BboxBase, Bbox, IdentityTransform, Transform, TransformedBbox,
def get_valid_values(self, attr):
    """
        Get the legal arguments for the setter associated with *attr*.

        This is done by querying the docstring of the setter for a line that
        begins with "ACCEPTS:" or ".. ACCEPTS:", and then by looking for a
        numpydoc-style documentation for the setter's first argument.
        """
    name = 'set_%s' % attr
    if not hasattr(self.o, name):
        raise AttributeError(f'{self.o} has no function {name}')
    func = getattr(self.o, name)
    docstring = inspect.getdoc(func)
    if docstring is None:
        return 'unknown'
    if docstring.startswith('Alias for '):
        return None
    match = self._get_valid_values_regex.search(docstring)
    if match is not None:
        return re.sub('\n *', ' ', match.group(1))
    param_name = func.__code__.co_varnames[1]
    match = re.search(f'(?m)^ *\\*?{param_name} : (.+)', docstring)
    if match:
        return match.group(1)
    return 'unknown'
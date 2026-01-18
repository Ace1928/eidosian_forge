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
def pprint_setters_rest(self, prop=None, leadingspace=4):
    """
        If *prop* is *None*, return a list of reST-formatted strings of all
        settable properties and their valid values.

        If *prop* is not *None*, it is a valid property name and that
        property will be returned as a string of "property : valid"
        values.
        """
    if leadingspace:
        pad = ' ' * leadingspace
    else:
        pad = ''
    if prop is not None:
        accepts = self.get_valid_values(prop)
        return f'{pad}{prop}: {accepts}'
    prop_and_qualnames = []
    for prop in sorted(self.get_setters()):
        for cls in self.o.__mro__:
            method = getattr(cls, f'set_{prop}', None)
            if method and method.__doc__ is not None:
                break
        else:
            method = getattr(self.o, f'set_{prop}')
        prop_and_qualnames.append((prop, f'{method.__module__}.{method.__qualname__}'))
    names = [self.aliased_name_rest(prop, target).replace('_base._AxesBase', 'Axes').replace('_axes.Axes', 'Axes') for prop, target in prop_and_qualnames]
    accepts = [self.get_valid_values(prop) for prop, _ in prop_and_qualnames]
    col0_len = max((len(n) for n in names))
    col1_len = max((len(a) for a in accepts))
    table_formatstr = pad + '   ' + '=' * col0_len + '   ' + '=' * col1_len
    return ['', pad + '.. table::', pad + '   :class: property-table', '', table_formatstr, pad + '   ' + 'Property'.ljust(col0_len) + '   ' + 'Description'.ljust(col1_len), table_formatstr, *[pad + '   ' + n.ljust(col0_len) + '   ' + a.ljust(col1_len) for n, a in zip(names, accepts)], table_formatstr, '']
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
def pprint_getters(self):
    """Return the getters and actual values as list of strings."""
    lines = []
    for name, val in sorted(self.properties().items()):
        if getattr(val, 'shape', ()) != () and len(val) > 6:
            s = str(val[:6]) + '...'
        else:
            s = str(val)
        s = s.replace('\n', ' ')
        if len(s) > 50:
            s = s[:50] + '...'
        name = self.aliased_name(name)
        lines.append(f'    {name} = {s}')
    return lines
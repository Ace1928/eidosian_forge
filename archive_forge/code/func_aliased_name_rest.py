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
def aliased_name_rest(self, s, target):
    """
        Return 'PROPNAME or alias' if *s* has an alias, else return 'PROPNAME',
        formatted for reST.

        For example, for the line markerfacecolor property, which has an
        alias, return 'markerfacecolor or mfc' and for the transform
        property, which does not, return 'transform'.
        """
    if target in self._NOT_LINKABLE:
        return f'``{s}``'
    aliases = ''.join((' or %s' % x for x in sorted(self.aliasd.get(s, []))))
    return f':meth:`{s} <{target}>`{aliases}'
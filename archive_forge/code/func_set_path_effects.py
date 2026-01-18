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
def set_path_effects(self, path_effects):
    """
        Set the path effects.

        Parameters
        ----------
        path_effects : list of `.AbstractPathEffect`
        """
    self._path_effects = path_effects
    self.stale = True
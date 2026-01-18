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
def set_sketch_params(self, scale=None, length=None, randomness=None):
    """
        Set the sketch parameters.

        Parameters
        ----------
        scale : float, optional
            The amplitude of the wiggle perpendicular to the source
            line, in pixels.  If scale is `None`, or not provided, no
            sketch filter will be provided.
        length : float, optional
             The length of the wiggle along the line, in pixels
             (default 128.0)
        randomness : float, optional
            The scale factor by which the length is shrunken or
            expanded (default 16.0)

            The PGF backend uses this argument as an RNG seed and not as
            described above. Using the same seed yields the same random shape.

            .. ACCEPTS: (scale: float, length: float, randomness: float)
        """
    if scale is None:
        self._sketch = None
    else:
        self._sketch = (scale, length or 128.0, randomness or 16.0)
    self.stale = True
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
def set_clip_on(self, b):
    """
        Set whether the artist uses clipping.

        When False, artists will be visible outside the Axes which
        can lead to unexpected results.

        Parameters
        ----------
        b : bool
        """
    self._clipon = b
    self.pchanged()
    self.stale = True
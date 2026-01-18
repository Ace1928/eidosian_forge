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
def set_clip_box(self, clipbox):
    """
        Set the artist's clip `.Bbox`.

        Parameters
        ----------
        clipbox : `~matplotlib.transforms.BboxBase` or None
            Will typically be created from a `.TransformedBbox`. For instance,
            ``TransformedBbox(Bbox([[0, 0], [1, 1]]), ax.transAxes)`` is the default
            clipping for an artist added to an Axes.

        """
    _api.check_isinstance((BboxBase, None), clipbox=clipbox)
    if clipbox != self.clipbox:
        self.clipbox = clipbox
        self.pchanged()
        self.stale = True
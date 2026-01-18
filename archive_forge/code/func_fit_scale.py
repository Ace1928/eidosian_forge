import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def fit_scale(self, rect, scale_min=0, scale_max=None, delta=0.001, verbose=False):
    """
        Finds smallest value `scale` in range `scale_min..scale_max` where
        `scale * rect` is large enough to contain the story `self`.

        Returns a `Story.FitResult` instance.

        :arg width:
            width of rect.
        :arg height:
            height of rect.
        :arg scale_min:
            Minimum scale to consider; must be >= 0.
        :arg scale_max:
            Maximum scale to consider, must be >= scale_min or `None` for
            infinite.
        :arg delta:
            Maximum error in returned scale.
        :arg verbose:
            If true we output diagnostics.
        """
    x0, y0, x1, y1 = rect
    width = x1 - x0
    height = y1 - y0

    def fn(scale):
        return Rect(x0, y0, x0 + scale * width, y0 + scale * height)
    return self.fit(fn, scale_min, scale_max, delta, verbose)
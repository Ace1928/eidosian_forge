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
def fit_width(self, height, width_min=0, width_max=None, origin=(0, 0), delta=0.001, verbose=False):
    """
        Finds smallest width in range `width_min..width_max` where a rect with size
        `(width, height)` is large enough to contain the story `self`.

        Returns a `Story.FitResult` instance.
        Returns a `FitResult` instance.

        :arg height:
            height of rect.
        :arg width_min:
            Minimum width to consider; must be >= 0.
        :arg width_max:
            Maximum width to consider, must be >= width_min or `None` for
            infinite.
        :arg origin:
            `(x0, y0)` of rect.
        :arg delta:
            Maximum error in returned width.
        :arg verbose:
            If true we output diagnostics.
        """
    x0, y0 = origin
    y1 = x0 + height

    def fn(width):
        return Rect(x0, y0, x0 + width, y1)
    return self.fit(fn, width_min, width_max, delta, verbose)
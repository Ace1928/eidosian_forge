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
def fit_height(self, width, height_min=0, height_max=None, origin=(0, 0), delta=0.001, verbose=False):
    """
        Finds smallest height in range `height_min..height_max` where a rect
        with size `(width, height)` is large enough to contain the story
        `self`.

        Returns a `Story.FitResult` instance.

        :arg width:
            width of rect.
        :arg height_min:
            Minimum height to consider; must be >= 0.
        :arg height_max:
            Maximum height to consider, must be >= height_min or `None` for
            infinite.
        :arg origin:
            `(x0, y0)` of rect.
        :arg delta:
            Maximum error in returned height.
        :arg verbose:
            If true we output diagnostics.
        """
    x0, y0 = origin
    x1 = x0 + width

    def fn(height):
        return Rect(x0, y0, x1, y0 + height)
    return self.fit(fn, height_min, height_max, delta, verbose)
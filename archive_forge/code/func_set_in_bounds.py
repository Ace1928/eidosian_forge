import collections
import copy
import datetime as dt
import glob
import inspect
import numbers
import os.path
import pathlib
import re
import sys
import typing
import warnings
from collections import OrderedDict
from contextlib import contextmanager
from .parameterized import (
from ._utils import (
def set_in_bounds(self, obj, val):
    """
        Set to the given value, but cropped to be within the legal bounds.
        All objects are accepted, and no exceptions will be raised.  See
        crop_to_bounds for details on how cropping is done.
        """
    if not callable(val):
        bounded_val = self.crop_to_bounds(val)
    else:
        bounded_val = val
    super().__set__(obj, bounded_val)
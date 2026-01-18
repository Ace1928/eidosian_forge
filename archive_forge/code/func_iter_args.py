import collections
from collections.abc import Iterable
import functools
import math
import warnings
from affine import Affine
import attr
import numpy as np
from rasterio.errors import WindowError, RasterioDeprecationWarning
from rasterio.transform import rowcol, guard_transform
def iter_args(function):
    """Decorator to allow function to take either ``*args`` or
    a single iterable which gets expanded to ``*args``.
    """

    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        if len(args) == 1 and isinstance(args[0], Iterable):
            return function(*args[0])
        else:
            return function(*args)
    return wrapper
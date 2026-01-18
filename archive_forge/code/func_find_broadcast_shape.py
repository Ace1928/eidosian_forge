import operator
import threading
import functools
import itertools
import contextlib
import collections
from ..autoray import (
from .draw import (
@functools.lru_cache(1024)
def find_broadcast_shape(xshape, yshape):
    xndim = len(xshape)
    yndim = len(yshape)
    if xndim < yndim:
        xshape = (1,) * (yndim - xndim) + xshape
    elif yndim < xndim:
        yshape = (1,) * (xndim - yndim) + yshape
    return tuple((max(d1, d2) for d1, d2 in zip(xshape, yshape)))
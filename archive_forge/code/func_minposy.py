import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
@property
def minposy(self):
    """
        The minimum positive value in the *y*-direction within the Bbox.

        This is useful when dealing with logarithmic scales and other scales
        where negative bounds result in floating point errors, and will be used
        as the minimum *y*-extent instead of *y0*.
        """
    return self._minpos[1]
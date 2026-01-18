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
def fully_contains(self, x, y):
    return self._bbox.fully_contains(*self._transform.inverted().transform((x, y)))
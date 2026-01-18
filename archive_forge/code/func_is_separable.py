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
def is_separable(self):
    mtx = self.get_matrix()
    return mtx[0, 1] == mtx[1, 0] == 0.0
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
def set_children(self, *children):
    """
        Set the children of the transform, to let the invalidation
        system know which transforms can invalidate this transform.
        Should be called from the constructor of any transforms that
        depend on other transforms.
        """
    id_self = id(self)
    for child in children:
        ref = weakref.ref(self, lambda _, pop=child._parents.pop, k=id_self: pop(k))
        child._parents[id_self] = ref
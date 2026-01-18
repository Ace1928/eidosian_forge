import os
import copy
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
import pytest
from pytest import raises as assert_raises
import scipy.spatial._qhull as qhull
from scipy.spatial import cKDTree as KDTree
from scipy.spatial import Voronoi
import itertools
def remap(x):
    if hasattr(x, '__len__'):
        return tuple({remap(y) for y in x})
    try:
        return vertex_map[x]
    except KeyError as e:
        message = f'incremental result has spurious vertex at {objx.vertices[x]!r}'
        raise AssertionError(message) from e
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
def assert_unordered_tuple_list_equal(a, b, tpl=tuple):
    if isinstance(a, np.ndarray):
        a = a.tolist()
    if isinstance(b, np.ndarray):
        b = b.tolist()
    a = list(map(tpl, a))
    a.sort()
    b = list(map(tpl, b))
    b.sort()
    assert_equal(a, b)
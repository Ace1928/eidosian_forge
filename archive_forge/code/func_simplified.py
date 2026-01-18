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
def simplified(x):
    items = set(map(sorted_tuple, x))
    if () in items:
        items.remove(())
    items = [x for x in items if len(x) > 1]
    items.sort()
    return items
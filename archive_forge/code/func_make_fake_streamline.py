import copy
import operator
import sys
import unittest
import warnings
from collections import defaultdict
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from ...testing import assert_arrays_equal, clear_and_catch_warnings
from .. import tractogram as module_tractogram
from ..tractogram import (
def make_fake_streamline(nb_points, data_per_point_shapes={}, data_for_streamline_shapes={}, rng=None):
    """Make a single streamline according to provided requirements."""
    if rng is None:
        rng = np.random.RandomState()
    streamline = rng.randn(nb_points, 3).astype('f4')
    data_per_point = {}
    for k, shape in data_per_point_shapes.items():
        data_per_point[k] = rng.randn(*(nb_points,) + shape).astype('f4')
    data_for_streamline = {}
    for k, shape in data_for_streamline.items():
        data_for_streamline[k] = rng.randn(*shape).astype('f4')
    return (streamline, data_per_point, data_for_streamline)
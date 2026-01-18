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
def test_creating_tractogram_item(self):
    rng = np.random.RandomState(42)
    streamline = rng.rand(rng.randint(10, 50), 3)
    colors = rng.rand(len(streamline), 3)
    mean_curvature = 1.11
    mean_color = np.array([0, 1, 0], dtype='f4')
    data_for_streamline = {'mean_curvature': mean_curvature, 'mean_color': mean_color}
    data_for_points = {'colors': colors}
    t = TractogramItem(streamline, data_for_streamline, data_for_points)
    assert len(t) == len(streamline)
    assert_array_equal(t.streamline, streamline)
    assert_array_equal(list(t), streamline)
    assert_array_equal(t.data_for_streamline['mean_curvature'], mean_curvature)
    assert_array_equal(t.data_for_streamline['mean_color'], mean_color)
    assert_array_equal(t.data_for_points['colors'], colors)
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
def make_dummy_streamline(nb_points):
    """Make the streamlines that have been used to create test data files."""
    if nb_points == 1:
        streamline = np.arange(1 * 3, dtype='f4').reshape((1, 3))
        data_per_point = {'fa': np.array([[0.2]], dtype='f4'), 'colors': np.array([(1, 0, 0)] * 1, dtype='f4')}
        data_for_streamline = {'mean_curvature': np.array([1.11], dtype='f4'), 'mean_torsion': np.array([1.22], dtype='f4'), 'mean_colors': np.array([1, 0, 0], dtype='f4')}
    elif nb_points == 2:
        streamline = np.arange(2 * 3, dtype='f4').reshape((2, 3))
        data_per_point = {'fa': np.array([[0.3], [0.4]], dtype='f4'), 'colors': np.array([(0, 1, 0)] * 2, dtype='f4')}
        data_for_streamline = {'mean_curvature': np.array([2.11], dtype='f4'), 'mean_torsion': np.array([2.22], dtype='f4'), 'mean_colors': np.array([0, 1, 0], dtype='f4')}
    elif nb_points == 5:
        streamline = np.arange(5 * 3, dtype='f4').reshape((5, 3))
        data_per_point = {'fa': np.array([[0.5], [0.6], [0.6], [0.7], [0.8]], dtype='f4'), 'colors': np.array([(0, 0, 1)] * 5, dtype='f4')}
        data_for_streamline = {'mean_curvature': np.array([3.11], dtype='f4'), 'mean_torsion': np.array([3.22], dtype='f4'), 'mean_colors': np.array([0, 0, 1], dtype='f4')}
    return (streamline, data_per_point, data_for_streamline)
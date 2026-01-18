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
def check_tractogram_item(tractogram_item, streamline, data_for_streamline={}, data_for_points={}):
    assert_array_equal(tractogram_item.streamline, streamline)
    assert len(tractogram_item.data_for_streamline) == len(data_for_streamline)
    for key in data_for_streamline.keys():
        assert_array_equal(tractogram_item.data_for_streamline[key], data_for_streamline[key])
    assert len(tractogram_item.data_for_points) == len(data_for_points)
    for key in data_for_points.keys():
        assert_arrays_equal(tractogram_item.data_for_points[key], data_for_points[key])
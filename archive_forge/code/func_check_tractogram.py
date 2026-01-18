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
def check_tractogram(tractogram, streamlines=[], data_per_streamline={}, data_per_point={}):
    streamlines = list(streamlines)
    assert len(tractogram) == len(streamlines)
    assert_arrays_equal(tractogram.streamlines, streamlines)
    [t for t in tractogram]
    assert len(tractogram.data_per_streamline) == len(data_per_streamline)
    for key in data_per_streamline.keys():
        assert_arrays_equal(tractogram.data_per_streamline[key], data_per_streamline[key])
    assert len(tractogram.data_per_point) == len(data_per_point)
    for key in data_per_point.keys():
        assert_arrays_equal(tractogram.data_per_point[key], data_per_point[key])
import itertools
import sys
import warnings
from io import BytesIO
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal, assert_array_equal
from nibabel.tmpdirs import InTemporaryDirectory
from ... import load
from ...fileholders import FileHolder
from ...nifti1 import data_type_codes
from ...testing import get_test_data
from .. import (
from .test_parse_gifti_fast import (
def test_agg_data():
    surf_gii_img = load(get_test_data('gifti', 'ascii.gii'))
    func_gii_img = load(get_test_data('gifti', 'task.func.gii'))
    shape_gii_img = load(get_test_data('gifti', 'rh.shape.curv.gii'))
    point_data = surf_gii_img.get_arrays_from_intent('pointset')[0].data
    triangle_data = surf_gii_img.get_arrays_from_intent('triangle')[0].data
    func_da = func_gii_img.get_arrays_from_intent('time series')
    func_data = np.column_stack(tuple((da.data for da in func_da)))
    shape_data = shape_gii_img.get_arrays_from_intent('shape')[0].data
    assert surf_gii_img.agg_data() == (point_data, triangle_data)
    assert_array_equal(func_gii_img.agg_data(), func_data)
    assert_array_equal(shape_gii_img.agg_data(), shape_data)
    assert_array_equal(surf_gii_img.agg_data('pointset'), point_data)
    assert_array_equal(surf_gii_img.agg_data('triangle'), triangle_data)
    assert_array_equal(func_gii_img.agg_data('time series'), func_data)
    assert_array_equal(shape_gii_img.agg_data('shape'), shape_data)
    assert surf_gii_img.agg_data('time series') == ()
    assert func_gii_img.agg_data('triangle') == ()
    assert shape_gii_img.agg_data('pointset') == ()
    assert surf_gii_img.agg_data(('pointset', 'triangle')) == (point_data, triangle_data)
    assert surf_gii_img.agg_data(('triangle', 'pointset')) == (triangle_data, point_data)
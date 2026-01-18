import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal
from scipy.spatial import distance
from skimage._shared._warnings import expected_warnings
from skimage.metrics import hausdorff_distance, hausdorff_pair
def test_gallery():
    shape = (60, 60)
    x_diamond = 30
    y_diamond = 30
    r = 10
    plt_x = [0, 1, 0, -1]
    plt_y = [1, 0, -1, 0]
    set_ax = [x_diamond + r * x for x in plt_x]
    set_ay = [y_diamond + r * y for y in plt_y]
    x_kite = 30
    y_kite = 30
    x_r = 15
    y_r = 20
    set_bx = [x_kite + x_r * x for x in plt_x]
    set_by = [y_kite + y_r * y for y in plt_y]
    coords_a = np.zeros(shape, dtype=bool)
    coords_b = np.zeros(shape, dtype=bool)
    for x, y in zip(set_ax, set_ay):
        coords_a[x, y] = True
    for x, y in zip(set_bx, set_by):
        coords_b[x, y] = True
    assert_almost_equal(hausdorff_distance(coords_a, coords_b), 10.0)
    hd_points = hausdorff_pair(coords_a, coords_b)
    assert np.equal(hd_points, ((30, 20), (30, 10))).all() or np.equal(hd_points, ((30, 40), (30, 50))).all()
    assert_almost_equal(hausdorff_distance(coords_a, coords_b, method='modified'), 7.5)
import warnings
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from ..affines import from_matvec, to_matvec
from ..orientations import (
from ..testing import deprecated_to, expires
def test_io_orientation():
    for shape in ((2, 3, 4), (20, 15, 7)):
        for in_arr, out_ornt in zip(IN_ARRS, OUT_ORNTS):
            ornt = io_orientation(in_arr)
            assert_array_equal(ornt, out_ornt)
            taff = inv_ornt_aff(ornt, shape)
            assert same_transform(taff, ornt, shape)
            for axno in range(3):
                arr = in_arr.copy()
                ex_ornt = out_ornt.copy()
                arr[:, axno] *= -1
                ex_ornt[axno, 1] *= -1
                ornt = io_orientation(arr)
                assert_array_equal(ornt, ex_ornt)
                taff = inv_ornt_aff(ornt, shape)
                assert same_transform(taff, ornt, shape)
    rzs = np.c_[np.diag([2, 3, 4, 5]), np.zeros((4, 3))]
    arr = from_matvec(rzs, [15, 16, 17, 18])
    ornt = io_orientation(arr)
    assert_array_equal(ornt, [[0, 1], [1, 1], [2, 1], [3, 1], [np.nan, np.nan], [np.nan, np.nan], [np.nan, np.nan]])
    def_aff = np.array([[1.0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    fail_tol = np.array([[0, 1], [np.nan, np.nan], [2, 1]])
    pass_tol = np.array([[0, 1], [1, 1], [2, 1]])
    eps = np.finfo(float).eps
    for y_val, has_y in ((0, False), (eps, False), (eps * 5, False), (eps * 10, True)):
        def_aff[1, 1] = y_val
        res = pass_tol if has_y else fail_tol
        assert_array_equal(io_orientation(def_aff), res)
    def_aff[1, 1] = eps
    assert_array_equal(io_orientation(def_aff, tol=0), pass_tol)
    def_aff[1, 1] = eps * 10
    assert_array_equal(io_orientation(def_aff, tol=1e-05), fail_tol)
    mat, vec = to_matvec(def_aff)
    aff_extra_col = np.zeros((4, 5))
    aff_extra_col[-1, -1] = 1
    aff_extra_col[:3, :3] = mat
    aff_extra_col[:3, -1] = vec
    assert_array_equal(io_orientation(aff_extra_col, tol=1e-05), [[0, 1], [np.nan, np.nan], [2, 1], [np.nan, np.nan]])
    aff_extra_row = np.zeros((5, 4))
    aff_extra_row[-1, -1] = 1
    aff_extra_row[:3, :3] = mat
    aff_extra_row[:3, -1] = vec
    assert_array_equal(io_orientation(aff_extra_row, tol=1e-05), [[0, 1], [np.nan, np.nan], [2, 1]])
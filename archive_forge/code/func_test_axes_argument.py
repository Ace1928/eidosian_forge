from numpy.testing import (assert_, assert_equal, assert_array_almost_equal,
import pytest
from pytest import raises as assert_raises
from scipy.fft._pocketfft import (ifft, fft, fftn, ifftn,
from numpy import (arange, array, asarray, zeros, dot, exp, pi,
import numpy as np
import numpy.fft
from numpy.random import rand
def test_axes_argument(self):
    plane1 = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    plane2 = [[10, 11, 12], [13, 14, 15], [16, 17, 18]]
    plane3 = [[19, 20, 21], [22, 23, 24], [25, 26, 27]]
    ki_plane1 = [[1, 2, 3], [10, 11, 12], [19, 20, 21]]
    ki_plane2 = [[4, 5, 6], [13, 14, 15], [22, 23, 24]]
    ki_plane3 = [[7, 8, 9], [16, 17, 18], [25, 26, 27]]
    jk_plane1 = [[1, 10, 19], [4, 13, 22], [7, 16, 25]]
    jk_plane2 = [[2, 11, 20], [5, 14, 23], [8, 17, 26]]
    jk_plane3 = [[3, 12, 21], [6, 15, 24], [9, 18, 27]]
    kj_plane1 = [[1, 4, 7], [10, 13, 16], [19, 22, 25]]
    kj_plane2 = [[2, 5, 8], [11, 14, 17], [20, 23, 26]]
    kj_plane3 = [[3, 6, 9], [12, 15, 18], [21, 24, 27]]
    ij_plane1 = [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    ij_plane2 = [[10, 13, 16], [11, 14, 17], [12, 15, 18]]
    ij_plane3 = [[19, 22, 25], [20, 23, 26], [21, 24, 27]]
    ik_plane1 = [[1, 10, 19], [2, 11, 20], [3, 12, 21]]
    ik_plane2 = [[4, 13, 22], [5, 14, 23], [6, 15, 24]]
    ik_plane3 = [[7, 16, 25], [8, 17, 26], [9, 18, 27]]
    ijk_space = [jk_plane1, jk_plane2, jk_plane3]
    ikj_space = [kj_plane1, kj_plane2, kj_plane3]
    jik_space = [ik_plane1, ik_plane2, ik_plane3]
    jki_space = [ki_plane1, ki_plane2, ki_plane3]
    kij_space = [ij_plane1, ij_plane2, ij_plane3]
    x = array([plane1, plane2, plane3])
    assert_array_almost_equal(fftn(x), fftn(x, axes=(-3, -2, -1)))
    assert_array_almost_equal(fftn(x), fftn(x, axes=(0, 1, 2)))
    assert_array_almost_equal(fftn(x, axes=(0, 2)), fftn(x, axes=(0, -1)))
    y = fftn(x, axes=(2, 1, 0))
    assert_array_almost_equal(swapaxes(y, -1, -3), fftn(ijk_space))
    y = fftn(x, axes=(2, 0, 1))
    assert_array_almost_equal(swapaxes(swapaxes(y, -1, -3), -1, -2), fftn(ikj_space))
    y = fftn(x, axes=(1, 2, 0))
    assert_array_almost_equal(swapaxes(swapaxes(y, -1, -3), -3, -2), fftn(jik_space))
    y = fftn(x, axes=(1, 0, 2))
    assert_array_almost_equal(swapaxes(y, -2, -3), fftn(jki_space))
    y = fftn(x, axes=(0, 2, 1))
    assert_array_almost_equal(swapaxes(y, -2, -1), fftn(kij_space))
    y = fftn(x, axes=(-2, -1))
    assert_array_almost_equal(fftn(plane1), y[0])
    assert_array_almost_equal(fftn(plane2), y[1])
    assert_array_almost_equal(fftn(plane3), y[2])
    y = fftn(x, axes=(1, 2))
    assert_array_almost_equal(fftn(plane1), y[0])
    assert_array_almost_equal(fftn(plane2), y[1])
    assert_array_almost_equal(fftn(plane3), y[2])
    y = fftn(x, axes=(-3, -2))
    assert_array_almost_equal(fftn(x[:, :, 0]), y[:, :, 0])
    assert_array_almost_equal(fftn(x[:, :, 1]), y[:, :, 1])
    assert_array_almost_equal(fftn(x[:, :, 2]), y[:, :, 2])
    y = fftn(x, axes=(-3, -1))
    assert_array_almost_equal(fftn(x[:, 0, :]), y[:, 0, :])
    assert_array_almost_equal(fftn(x[:, 1, :]), y[:, 1, :])
    assert_array_almost_equal(fftn(x[:, 2, :]), y[:, 2, :])
    y = fftn(x, axes=(-1, -2))
    assert_array_almost_equal(fftn(ij_plane1), swapaxes(y[0], -2, -1))
    assert_array_almost_equal(fftn(ij_plane2), swapaxes(y[1], -2, -1))
    assert_array_almost_equal(fftn(ij_plane3), swapaxes(y[2], -2, -1))
    y = fftn(x, axes=(-1, -3))
    assert_array_almost_equal(fftn(ik_plane1), swapaxes(y[:, 0, :], -1, -2))
    assert_array_almost_equal(fftn(ik_plane2), swapaxes(y[:, 1, :], -1, -2))
    assert_array_almost_equal(fftn(ik_plane3), swapaxes(y[:, 2, :], -1, -2))
    y = fftn(x, axes=(-2, -3))
    assert_array_almost_equal(fftn(jk_plane1), swapaxes(y[:, :, 0], -1, -2))
    assert_array_almost_equal(fftn(jk_plane2), swapaxes(y[:, :, 1], -1, -2))
    assert_array_almost_equal(fftn(jk_plane3), swapaxes(y[:, :, 2], -1, -2))
    y = fftn(x, axes=(-1,))
    for i in range(3):
        for j in range(3):
            assert_array_almost_equal(fft(x[i, j, :]), y[i, j, :])
    y = fftn(x, axes=(-2,))
    for i in range(3):
        for j in range(3):
            assert_array_almost_equal(fft(x[i, :, j]), y[i, :, j])
    y = fftn(x, axes=(0,))
    for i in range(3):
        for j in range(3):
            assert_array_almost_equal(fft(x[:, i, j]), y[:, i, j])
    y = fftn(x, axes=())
    assert_array_almost_equal(y, x)
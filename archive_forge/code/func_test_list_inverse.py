from __future__ import division
from uncertainties import unumpy, ufloat
from uncertainties.unumpy.test_unumpy import arrays_close
def test_list_inverse():
    """Test of the inversion of a square matrix"""
    mat_list = [[1, 1], [1, 0]]
    mat_list_inv = unumpy.ulinalg.inv(mat_list)
    mat_matrix = numpy.asmatrix(mat_list)
    assert isinstance(unumpy.ulinalg.inv(mat_matrix), type(numpy.linalg.inv(mat_matrix)))
    mat_list_inv_numpy = numpy.linalg.inv(mat_list)
    assert type(mat_list_inv) == type(mat_list_inv_numpy)
    assert not isinstance(mat_list_inv, unumpy.matrix)
    assert isinstance(mat_list_inv[1, 1], float)
    assert mat_list_inv[1, 1] == -1
    x = ufloat(1, 0.1)
    y = ufloat(2, 0.1)
    mat = unumpy.matrix([[x, x], [y, 0]])
    assert arrays_close(unumpy.ulinalg.inv(mat), mat.I)
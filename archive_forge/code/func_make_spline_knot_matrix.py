import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from scipy import ndimage
def make_spline_knot_matrix(n, order, mode='mirror'):
    """Matrix to invert to find the spline coefficients."""
    knot_values = get_spline_knot_values(order)
    matrix = np.zeros((n, n))
    for diag, knot_value in enumerate(knot_values):
        indices = np.arange(diag, n)
        if diag == 0:
            matrix[indices, indices] = knot_value
        else:
            matrix[indices, indices - diag] = knot_value
            matrix[indices - diag, indices] = knot_value
    knot_values_sum = knot_values[0] + 2 * sum(knot_values[1:])
    if mode == 'mirror':
        start, step = (1, 1)
    elif mode == 'reflect':
        start, step = (0, 1)
    elif mode == 'grid-wrap':
        start, step = (-1, -1)
    else:
        raise ValueError(f'unsupported mode {mode}')
    for row in range(len(knot_values) - 1):
        for idx, knot_value in enumerate(knot_values[row + 1:]):
            matrix[row, start + step * idx] += knot_value
            matrix[-row - 1, -start - 1 - step * idx] += knot_value
    return matrix / knot_values_sum
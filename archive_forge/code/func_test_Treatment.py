from __future__ import print_function
import sys
import six
import numpy as np
from patsy import PatsyError
from patsy.util import (repr_pretty_delegate, repr_pretty_impl,
def test_Treatment():
    t1 = Treatment()
    matrix = t1.code_with_intercept(['a', 'b', 'c'])
    assert matrix.column_suffixes == ['[a]', '[b]', '[c]']
    assert np.allclose(matrix.matrix, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    matrix = t1.code_without_intercept(['a', 'b', 'c'])
    assert matrix.column_suffixes == ['[T.b]', '[T.c]']
    assert np.allclose(matrix.matrix, [[0, 0], [1, 0], [0, 1]])
    matrix = Treatment(reference=1).code_without_intercept(['a', 'b', 'c'])
    assert matrix.column_suffixes == ['[T.a]', '[T.c]']
    assert np.allclose(matrix.matrix, [[1, 0], [0, 0], [0, 1]])
    matrix = Treatment(reference=-2).code_without_intercept(['a', 'b', 'c'])
    assert matrix.column_suffixes == ['[T.a]', '[T.c]']
    assert np.allclose(matrix.matrix, [[1, 0], [0, 0], [0, 1]])
    matrix = Treatment(reference='b').code_without_intercept(['a', 'b', 'c'])
    assert matrix.column_suffixes == ['[T.a]', '[T.c]']
    assert np.allclose(matrix.matrix, [[1, 0], [0, 0], [0, 1]])
    matrix = Treatment().code_without_intercept([2, 1, 0])
    assert matrix.column_suffixes == ['[T.1]', '[T.0]']
    assert np.allclose(matrix.matrix, [[0, 0], [1, 0], [0, 1]])
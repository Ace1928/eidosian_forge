from __future__ import print_function
import re
import six
import numpy as np
from patsy import PatsyError
from patsy.origin import Origin
from patsy.util import (atleast_2d_column_default,
from patsy.infix_parser import Token, Operator, infix_parse
from patsy.parse_formula import _parsing_error_test
def test_linear_constraint():
    import pytest
    from patsy.compat import OrderedDict
    t = _check_lincon
    t(LinearConstraint(['a', 'b'], [2, 3]), ['a', 'b'], [[2, 3]], [[0]])
    pytest.raises(ValueError, linear_constraint, LinearConstraint(['b', 'a'], [2, 3]), ['a', 'b'])
    t({'a': 2}, ['a', 'b'], [[1, 0]], [[2]])
    t(OrderedDict([('a', 2), ('b', 3)]), ['a', 'b'], [[1, 0], [0, 1]], [[2], [3]])
    t(OrderedDict([('a', 2), ('b', 3)]), ['b', 'a'], [[0, 1], [1, 0]], [[2], [3]])
    t({0: 2}, ['a', 'b'], [[1, 0]], [[2]])
    t(OrderedDict([(0, 2), (1, 3)]), ['a', 'b'], [[1, 0], [0, 1]], [[2], [3]])
    t(OrderedDict([('a', 2), (1, 3)]), ['a', 'b'], [[1, 0], [0, 1]], [[2], [3]])
    pytest.raises(ValueError, linear_constraint, {'q': 1}, ['a', 'b'])
    pytest.raises(ValueError, linear_constraint, {'a': 1, 0: 2}, ['a', 'b'])
    t(np.array([2, 3]), ['a', 'b'], [[2, 3]], [[0]])
    t(np.array([[2, 3], [4, 5]]), ['a', 'b'], [[2, 3], [4, 5]], [[0], [0]])
    t('a = 2', ['a', 'b'], [[1, 0]], [[2]])
    t('a - 2', ['a', 'b'], [[1, 0]], [[2]])
    t('a + 1 = 3', ['a', 'b'], [[1, 0]], [[2]])
    t('a + b = 3', ['a', 'b'], [[1, 1]], [[3]])
    t('a = 2, b = 3', ['a', 'b'], [[1, 0], [0, 1]], [[2], [3]])
    t('b = 3, a = 2', ['a', 'b'], [[0, 1], [1, 0]], [[3], [2]])
    t(['a = 2', 'b = 3'], ['a', 'b'], [[1, 0], [0, 1]], [[2], [3]])
    pytest.raises(ValueError, linear_constraint, ['a', {'b': 0}], ['a', 'b'])
    t('2 * (a + b/3) + b + 2*3/4 = 1 + 2*3', ['a', 'b'], [[2, 2.0 / 3 + 1]], [[7 - 6.0 / 4]])
    t('+2 * -a', ['a', 'b'], [[-2, 0]], [[0]])
    t('a - b, a + b = 2', ['a', 'b'], [[1, -1], [1, 1]], [[0], [2]])
    t('a = 1, a = 2, a = 3', ['a', 'b'], [[1, 0], [1, 0], [1, 0]], [[1], [2], [3]])
    t('a * 2', ['a', 'b'], [[2, 0]], [[0]])
    t('-a = 1', ['a', 'b'], [[-1, 0]], [[1]])
    t('(2 + a - a) * b', ['a', 'b'], [[0, 2]], [[0]])
    t('a = 1 = b', ['a', 'b'], [[1, 0], [0, -1]], [[1], [-1]])
    t('a = (1 = b)', ['a', 'b'], [[0, -1], [1, 0]], [[-1], [1]])
    t('a = 1, a = b = c', ['a', 'b', 'c'], [[1, 0, 0], [1, -1, 0], [0, 1, -1]], [[1], [0], [0]])
    t('a + 1 = 2', ['a', 'a + 1'], [[0, 1]], [[2]])
    t(([10, 20], [30]), ['a', 'b'], [[10, 20]], [[30]])
    t(([[10, 20], [20, 40]], [[30], [35]]), ['a', 'b'], [[10, 20], [20, 40]], [[30], [35]])
    pytest.raises(ValueError, linear_constraint, ([1, 0], [0], [0]), ['a', 'b'])
    pytest.raises(ValueError, linear_constraint, ([1, 0],), ['a', 'b'])
    t([10, 20], ['a', 'b'], [[10, 20]], [[0]])
    t([[10, 20], [20, 40]], ['a', 'b'], [[10, 20], [20, 40]], [[0], [0]])
    t(np.array([10, 20]), ['a', 'b'], [[10, 20]], [[0]])
    t(np.array([[10, 20], [20, 40]]), ['a', 'b'], [[10, 20], [20, 40]], [[0], [0]])
    pytest.raises(ValueError, linear_constraint, None, ['a', 'b'])
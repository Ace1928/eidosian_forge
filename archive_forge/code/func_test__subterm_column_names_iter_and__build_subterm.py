import itertools
import six
import numpy as np
from patsy import PatsyError
from patsy.categorical import (guess_categorical,
from patsy.util import (atleast_2d_column_default,
from patsy.design_info import (DesignMatrix, DesignInfo,
from patsy.redundancy import pick_contrasts_for_term
from patsy.eval import EvalEnvironment
from patsy.contrasts import code_contrast_matrix, Treatment
from patsy.compat import OrderedDict
from patsy.missing import NAAction
def test__subterm_column_names_iter_and__build_subterm():
    import pytest
    from patsy.contrasts import ContrastMatrix
    from patsy.categorical import C
    f1 = _MockFactor('f1')
    f2 = _MockFactor('f2')
    f3 = _MockFactor('f3')
    contrast = ContrastMatrix(np.array([[0, 0.5], [3, 0]]), ['[c1]', '[c2]'])
    factor_infos1 = {f1: FactorInfo(f1, 'numerical', {}, num_columns=1, categories=None), f2: FactorInfo(f2, 'categorical', {}, num_columns=None, categories=['a', 'b']), f3: FactorInfo(f3, 'numerical', {}, num_columns=1, categories=None)}
    contrast_matrices = {f2: contrast}
    subterm1 = SubtermInfo([f1, f2, f3], contrast_matrices, 2)
    assert list(_subterm_column_names_iter(factor_infos1, subterm1)) == ['f1:f2[c1]:f3', 'f1:f2[c2]:f3']
    mat = np.empty((3, 2))
    _build_subterm(subterm1, factor_infos1, {f1: atleast_2d_column_default([1, 2, 3]), f2: np.asarray([0, 0, 1]), f3: atleast_2d_column_default([7.5, 2, -12])}, mat)
    assert np.allclose(mat, [[0, 0.5 * 1 * 7.5], [0, 0.5 * 2 * 2], [3 * 3 * -12, 0]])
    pytest.raises(PatsyError, _build_subterm, subterm1, factor_infos1, {f1: atleast_2d_column_default([1, 2, 3]), f2: np.asarray([0, -1, 1]), f3: atleast_2d_column_default([7.5, 2, -12])}, mat)
    factor_infos2 = dict(factor_infos1)
    factor_infos2[f1] = FactorInfo(f1, 'numerical', {}, num_columns=2, categories=None)
    subterm2 = SubtermInfo([f1, f2, f3], contrast_matrices, 4)
    assert list(_subterm_column_names_iter(factor_infos2, subterm2)) == ['f1[0]:f2[c1]:f3', 'f1[1]:f2[c1]:f3', 'f1[0]:f2[c2]:f3', 'f1[1]:f2[c2]:f3']
    mat2 = np.empty((3, 4))
    _build_subterm(subterm2, factor_infos2, {f1: atleast_2d_column_default([[1, 2], [3, 4], [5, 6]]), f2: np.asarray([0, 0, 1]), f3: atleast_2d_column_default([7.5, 2, -12])}, mat2)
    assert np.allclose(mat2, [[0, 0, 0.5 * 1 * 7.5, 0.5 * 2 * 7.5], [0, 0, 0.5 * 3 * 2, 0.5 * 4 * 2], [3 * 5 * -12, 3 * 6 * -12, 0, 0]])
    subterm_int = SubtermInfo([], {}, 1)
    assert list(_subterm_column_names_iter({}, subterm_int)) == ['Intercept']
    mat3 = np.empty((3, 1))
    _build_subterm(subterm_int, {}, {f1: [1, 2, 3], f2: [1, 2, 3], f3: [1, 2, 3]}, mat3)
    assert np.allclose(mat3, 1)
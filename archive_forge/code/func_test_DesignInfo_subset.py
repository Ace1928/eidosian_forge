from __future__ import print_function
import six
import numpy as np
import pytest
from patsy import PatsyError
from patsy.util import (atleast_2d_column_default,
from patsy.desc import Term, INTERCEPT
from patsy.build import *
from patsy.categorical import C
from patsy.user_util import balanced, LookupFactor
from patsy.design_info import DesignMatrix, DesignInfo
def test_DesignInfo_subset():
    all_data = {'x': [1, 2], 'y': [[3.1, 3.2], [4.1, 4.2]], 'z': [5, 6]}
    all_terms = make_termlist('x', 'y', 'z')

    def iter_maker():
        yield all_data
    all_builder = design_matrix_builders([all_terms], iter_maker, 0)[0]
    full_matrix = build_design_matrices([all_builder], all_data)[0]

    def t(which_terms, variables, columns):
        sub_design_info = all_builder.subset(which_terms)
        sub_data = {}
        for variable in variables:
            sub_data[variable] = all_data[variable]
        sub_matrix = build_design_matrices([sub_design_info], sub_data)[0]
        sub_full_matrix = full_matrix[:, columns]
        if not isinstance(which_terms, six.string_types):
            assert len(which_terms) == len(sub_design_info.terms)
        assert np.array_equal(sub_matrix, sub_full_matrix)
    t('~ 0 + x + y + z', ['x', 'y', 'z'], slice(None))
    t(['x', 'y', 'z'], ['x', 'y', 'z'], slice(None))
    if not six.PY3:
        t([unicode('x'), unicode('y'), unicode('z')], ['x', 'y', 'z'], slice(None))
    t(all_terms, ['x', 'y', 'z'], slice(None))
    t([all_terms[0], 'y', all_terms[2]], ['x', 'y', 'z'], slice(None))
    t('~ 0 + x + z', ['x', 'z'], [0, 3])
    t(['x', 'z'], ['x', 'z'], [0, 3])
    if not six.PY3:
        t([unicode('x'), unicode('z')], ['x', 'z'], [0, 3])
    t([all_terms[0], all_terms[2]], ['x', 'z'], [0, 3])
    t([all_terms[0], 'z'], ['x', 'z'], [0, 3])
    t('~ 0 + z + x', ['x', 'z'], [3, 0])
    t(['z', 'x'], ['x', 'z'], [3, 0])
    t([six.text_type('z'), six.text_type('x')], ['x', 'z'], [3, 0])
    t([all_terms[2], all_terms[0]], ['x', 'z'], [3, 0])
    t([all_terms[2], 'x'], ['x', 'z'], [3, 0])
    t('~ 0 + y', ['y'], [1, 2])
    t(['y'], ['y'], [1, 2])
    t([six.text_type('y')], ['y'], [1, 2])
    t([all_terms[1]], ['y'], [1, 2])
    pytest.raises(PatsyError, all_builder.subset, 'a ~ a')
    pytest.raises(KeyError, all_builder.subset, '~ asdf')
    pytest.raises(KeyError, all_builder.subset, ['asdf'])
    pytest.raises(KeyError, all_builder.subset, [Term(['asdf'])])
    min_di = DesignInfo(['a', 'b', 'c'])
    min_di_subset = min_di.subset(['c', 'a'])
    assert min_di_subset.column_names == ['c', 'a']
    assert min_di_subset.terms is None
import numpy as np
import pytest
from opt_einsum import contract, contract_expression, contract_path, helpers
from opt_einsum.paths import linear_to_ssa, ssa_to_linear
@pytest.mark.parametrize('string', tests)
@pytest.mark.parametrize('optimize', all_optimizers)
@pytest.mark.parametrize('use_blas', [False, True])
@pytest.mark.parametrize('out_spec', [False, True])
def test_contract_expressions(string, optimize, use_blas, out_spec):
    views = helpers.build_views(string)
    shapes = [view.shape for view in views]
    expected = contract(string, *views, optimize=False, use_blas=False)
    expr = contract_expression(string, *shapes, optimize=optimize, use_blas=use_blas)
    if out_spec and '->' in string and (string[-2:] != '->'):
        out, = helpers.build_views(string.split('->')[1])
        expr(*views, out=out)
    else:
        out = expr(*views)
    assert np.allclose(out, expected)
    assert string in expr.__repr__()
    assert string in expr.__str__()
import numpy as np
import pytest
from opt_einsum import (backends, contract, contract_expression, helpers, sharing)
from opt_einsum.contract import Shaped, infer_backend, parse_backend
@pytest.mark.skipif(not found_jax, reason='jax not installed.')
@pytest.mark.parametrize('string', tests)
def test_jax(string):
    views = helpers.build_views(string)
    ein = contract(string, *views, optimize=False, use_blas=False)
    shps = [v.shape for v in views]
    expr = contract_expression(string, *shps, optimize=True)
    opt = expr(*views, backend='jax')
    assert np.allclose(ein, opt)
    assert isinstance(opt, np.ndarray)
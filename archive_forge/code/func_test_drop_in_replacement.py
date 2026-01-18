import numpy as np
import pytest
from opt_einsum import contract, contract_expression, contract_path, helpers
from opt_einsum.paths import linear_to_ssa, ssa_to_linear
@pytest.mark.parametrize('string', tests)
def test_drop_in_replacement(string):
    views = helpers.build_views(string)
    opt = contract(string, *views)
    assert np.allclose(opt, np.einsum(string, *views))
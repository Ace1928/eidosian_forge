import numpy as np
import pytest
from opt_einsum import contract, contract_expression
def test_broadcasting_contraction4():
    a = np.arange(64).reshape(2, 4, 8)
    ein = contract('obk,ijk->ioj', a, a, optimize=False)
    opt = contract('obk,ijk->ioj', a, a, optimize=True)
    assert np.allclose(ein, opt)
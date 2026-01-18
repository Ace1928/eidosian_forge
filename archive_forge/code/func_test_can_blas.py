import numpy as np
import pytest
from opt_einsum import blas, contract, helpers
@pytest.mark.parametrize('inp,benchmark', blas_tests)
def test_can_blas(inp, benchmark):
    result = blas.can_blas(*inp)
    assert result == benchmark
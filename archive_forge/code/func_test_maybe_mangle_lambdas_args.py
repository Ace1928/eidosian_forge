import numpy as np
import pytest
from pandas.core.apply import (
def test_maybe_mangle_lambdas_args():
    func = {'A': [lambda x, a, b=1: (0, a, b), lambda x: 1]}
    result = maybe_mangle_lambdas(func)
    assert result['A'][0].__name__ == '<lambda_0>'
    assert result['A'][1].__name__ == '<lambda_1>'
    assert func['A'][0](0, 1) == (0, 1, 1)
    assert func['A'][0](0, 1, 2) == (0, 1, 2)
    assert func['A'][0](0, 2, b=3) == (0, 2, 3)
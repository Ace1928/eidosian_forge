import numpy as np
import pytest
from pandas.core.apply import (
def test_maybe_mangle_lambdas():
    func = {'A': [lambda x: 0, lambda x: 1]}
    result = maybe_mangle_lambdas(func)
    assert result['A'][0].__name__ == '<lambda_0>'
    assert result['A'][1].__name__ == '<lambda_1>'
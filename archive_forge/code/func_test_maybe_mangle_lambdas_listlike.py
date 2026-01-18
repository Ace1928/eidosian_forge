import numpy as np
import pytest
from pandas.core.apply import (
def test_maybe_mangle_lambdas_listlike():
    aggfuncs = [lambda x: 1, lambda x: 2]
    result = maybe_mangle_lambdas(aggfuncs)
    assert result[0].__name__ == '<lambda_0>'
    assert result[1].__name__ == '<lambda_1>'
    assert aggfuncs[0](None) == result[0](None)
    assert aggfuncs[1](None) == result[1](None)
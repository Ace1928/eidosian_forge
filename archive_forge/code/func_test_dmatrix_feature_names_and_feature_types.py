import numpy as np
import pandas
import pytest
import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
import modin.experimental.xgboost as mxgb
import modin.pandas as pd
from modin.config import Engine
from modin.utils import try_cast_to_pandas
@pytest.mark.parametrize('data', [np.random.randn(5, 5), np.array([[1, 2], [3, 4]]), np.array([['a', 'b'], ['c', 'd']]), [[1, 2], [3, 4]], [['a', 'b'], ['c', 'd']]])
@pytest.mark.parametrize('feature_names', [list('abcdef'), ['a', 'b', 'c', 'd', 'd'], ['a', 'b', 'c', 'd', 'e<1'], list('abcde')])
@pytest.mark.parametrize('feature_types', [None, 'q', list('qiqiq')])
def test_dmatrix_feature_names_and_feature_types(data, feature_names, feature_types):
    check_dmatrix(data, feature_names=feature_names, feature_types=feature_types)
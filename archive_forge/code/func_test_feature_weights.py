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
def test_feature_weights():
    n_rows = 10
    n_cols = 50
    fw = rng.uniform(size=n_cols)
    X = rng.randn(n_rows, n_cols)
    dm = xgb.DMatrix(X)
    md_dm = mxgb.DMatrix(pd.DataFrame(X))
    dm.set_info(feature_weights=fw)
    md_dm.set_info(feature_weights=fw)
    np.testing.assert_allclose(dm.get_float_info('feature_weights'), md_dm.get_float_info('feature_weights'))
    dm.set_info(feature_weights=np.empty((0,)))
    md_dm.set_info(feature_weights=np.empty((0,)))
    assert dm.get_float_info('feature_weights').shape[0] == md_dm.get_float_info('feature_weights').shape[0] == 0
import numpy as np
import pandas as pd
from dask import array as da
from dask import dataframe as dd
from distributed import Client
import xgboost as xgb
from xgboost.testing.updater import get_basescore
def check_init_estimation_reg(tree_method: str, client: Client) -> None:
    """Test init estimation for regressor."""
    from sklearn.datasets import make_regression
    X, y = make_regression(n_samples=4096 * 2, n_features=32, random_state=1994)
    reg = xgb.XGBRegressor(n_estimators=1, max_depth=1, tree_method=tree_method)
    reg.fit(X, y)
    base_score = get_basescore(reg)
    dx = da.from_array(X).rechunk(chunks=(32, None))
    dy = da.from_array(y).rechunk(chunks=(32,))
    dreg = xgb.dask.DaskXGBRegressor(n_estimators=1, max_depth=1, tree_method=tree_method)
    dreg.client = client
    dreg.fit(dx, dy)
    dbase_score = get_basescore(dreg)
    np.testing.assert_allclose(base_score, dbase_score)
import json
from functools import partial, update_wrapper
from typing import Any, Dict, List
import numpy as np
import xgboost as xgb
import xgboost.testing as tm
def run_reg(X: np.ndarray, y: np.ndarray) -> None:
    reg = xgb.XGBRegressor(tree_method=tree_method, max_depth=1, n_estimators=1)
    reg.fit(X, y, eval_set=[(X, y)])
    base_score_0 = get_basescore(reg)
    score_0 = reg.evals_result()['validation_0']['rmse'][0]
    reg = xgb.XGBRegressor(tree_method=tree_method, max_depth=1, n_estimators=1, boost_from_average=0)
    reg.fit(X, y, eval_set=[(X, y)])
    base_score_1 = get_basescore(reg)
    score_1 = reg.evals_result()['validation_0']['rmse'][0]
    assert not np.isclose(base_score_0, base_score_1)
    assert score_0 < score_1
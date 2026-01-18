import json
from functools import partial, update_wrapper
from typing import Any, Dict, List
import numpy as np
import xgboost as xgb
import xgboost.testing as tm
def run_clf(X: np.ndarray, y: np.ndarray) -> None:
    clf = xgb.XGBClassifier(tree_method=tree_method, max_depth=1, n_estimators=1)
    clf.fit(X, y, eval_set=[(X, y)])
    base_score_0 = get_basescore(clf)
    score_0 = clf.evals_result()['validation_0']['logloss'][0]
    clf = xgb.XGBClassifier(tree_method=tree_method, max_depth=1, n_estimators=1, boost_from_average=0)
    clf.fit(X, y, eval_set=[(X, y)])
    base_score_1 = get_basescore(clf)
    score_1 = clf.evals_result()['validation_0']['logloss'][0]
    assert not np.isclose(base_score_0, base_score_1)
    assert score_0 < score_1
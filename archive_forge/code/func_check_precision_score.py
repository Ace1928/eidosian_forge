from typing import Dict, List
import numpy as np
import pytest
import xgboost as xgb
from xgboost.compat import concat
from xgboost.core import _parse_eval_str
def check_precision_score(tree_method: str) -> None:
    """Test for precision with ranking and classification."""
    datasets = pytest.importorskip('sklearn.datasets')
    X, y = datasets.make_classification(n_samples=1024, n_features=4, n_classes=2, random_state=2023)
    qid = np.zeros(shape=y.shape)
    ltr = xgb.XGBRanker(n_estimators=2, tree_method=tree_method)
    ltr.fit(X, y, qid=qid)
    X, y = datasets.make_classification(n_samples=512, n_features=4, n_classes=2, random_state=1994)
    ltr.set_params(eval_metric='pre@32')
    result = _parse_eval_str(ltr.get_booster().eval_set(evals=[(xgb.DMatrix(X, y), 'Xy')]))
    score_0 = result[1][1]
    X_list = []
    y_list = []
    n_query_groups = 3
    q_list: List[np.ndarray] = []
    for i in range(n_query_groups):
        X, y = datasets.make_classification(n_samples=512, n_features=4, n_classes=2, random_state=1994)
        X_list.append(X)
        y_list.append(y)
        q = np.full(shape=y.shape, fill_value=i, dtype=np.uint64)
        q_list.append(q)
    qid = concat(q_list)
    X = concat(X_list)
    y = concat(y_list)
    result = _parse_eval_str(ltr.get_booster().eval_set(evals=[(xgb.DMatrix(X, y, qid=qid), 'Xy')]))
    assert result[1][0].endswith('pre@32')
    score_1 = result[1][1]
    assert score_1 == score_0
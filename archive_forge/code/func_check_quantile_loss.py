import json
from functools import partial, update_wrapper
from typing import Any, Dict, List
import numpy as np
import xgboost as xgb
import xgboost.testing as tm
def check_quantile_loss(tree_method: str, weighted: bool) -> None:
    """Test for quantile loss."""
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_pinball_loss
    from xgboost.sklearn import _metric_decorator
    n_samples = 4096
    n_features = 8
    n_estimators = 8
    base_score = 0.0
    rng = np.random.RandomState(1994)
    X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=rng)
    if weighted:
        weight = rng.random(size=n_samples)
    else:
        weight = None
    Xy = xgb.QuantileDMatrix(X, y, weight=weight)
    alpha = np.array([0.1, 0.5])
    evals_result: Dict[str, Dict] = {}
    booster_multi = xgb.train({'objective': 'reg:quantileerror', 'tree_method': tree_method, 'quantile_alpha': alpha, 'base_score': base_score}, Xy, num_boost_round=n_estimators, evals=[(Xy, 'Train')], evals_result=evals_result)
    predt_multi = booster_multi.predict(Xy, strict_shape=True)
    assert tm.non_increasing(evals_result['Train']['quantile'])
    assert evals_result['Train']['quantile'][-1] < 20.0
    metrics = [_metric_decorator(update_wrapper(partial(mean_pinball_loss, sample_weight=weight, alpha=alpha[i]), mean_pinball_loss)) for i in range(alpha.size)]
    predts = np.empty(predt_multi.shape)
    for i in range(alpha.shape[0]):
        a = alpha[i]
        booster_i = xgb.train({'objective': 'reg:quantileerror', 'tree_method': tree_method, 'quantile_alpha': a, 'base_score': base_score}, Xy, num_boost_round=n_estimators, evals=[(Xy, 'Train')], custom_metric=metrics[i], evals_result=evals_result)
        assert tm.non_increasing(evals_result['Train']['quantile'])
        assert evals_result['Train']['quantile'][-1] < 30.0
        np.testing.assert_allclose(np.array(evals_result['Train']['quantile']), np.array(evals_result['Train']['mean_pinball_loss']), atol=1e-06, rtol=1e-06)
        predts[:, i] = booster_i.predict(Xy)
    for i in range(alpha.shape[0]):
        np.testing.assert_allclose(predts[:, i], predt_multi[:, i])
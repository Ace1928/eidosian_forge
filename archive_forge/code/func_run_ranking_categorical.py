from types import ModuleType
from typing import Any
import numpy as np
import pytest
import xgboost as xgb
from xgboost import testing as tm
def run_ranking_categorical(device: str) -> None:
    """Test LTR with categorical features."""
    from sklearn.model_selection import cross_val_score
    X, y = tm.make_categorical(n_samples=512, n_features=10, n_categories=3, onehot=False)
    rng = np.random.default_rng(1994)
    qid = rng.choice(3, size=y.shape[0])
    qid = np.sort(qid)
    X['qid'] = qid
    ltr = xgb.XGBRanker(enable_categorical=True, device=device)
    ltr.fit(X, y)
    score = ltr.score(X, y)
    assert score > 0.9
    ltr = xgb.XGBRanker(enable_categorical=True, device=device)
    scores = cross_val_score(ltr, X, y)
    for s in scores:
        assert s > 0.7
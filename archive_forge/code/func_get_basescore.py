import json
from functools import partial, update_wrapper
from typing import Any, Dict, List
import numpy as np
import xgboost as xgb
import xgboost.testing as tm
def get_basescore(model: xgb.XGBModel) -> float:
    """Get base score from an XGBoost sklearn estimator."""
    base_score = float(json.loads(model.get_booster().save_config())['learner']['learner_model_param']['base_score'])
    return base_score
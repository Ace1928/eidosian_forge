from types import ModuleType
from typing import Any
import numpy as np
import pytest
import xgboost as xgb
from xgboost import testing as tm
def neg_mse(*args: Any, **kwargs: Any) -> float:
    return -float(mean_squared_error(*args, **kwargs))
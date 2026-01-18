import warnings
from typing import Any, List, Optional, Type, Union
import numpy as np
from pyspark import keyword_only
from pyspark.ml.param import Param, Params
from pyspark.ml.param.shared import HasProbabilityCol, HasRawPredictionCol
from xgboost import XGBClassifier, XGBRanker, XGBRegressor
from .core import (  # type: ignore
from .utils import get_class_name
def param_value_converter(v: Any) -> Any:
    if isinstance(v, np.generic):
        return np.array(v).item()
    if isinstance(v, dict):
        return {k: param_value_converter(nv) for k, nv in v.items()}
    if isinstance(v, list):
        return [param_value_converter(nv) for nv in v]
    return v
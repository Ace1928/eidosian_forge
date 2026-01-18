import warnings
from typing import Any, List, Optional, Type, Union
import numpy as np
from pyspark import keyword_only
from pyspark.ml.param import Param, Params
from pyspark.ml.param.shared import HasProbabilityCol, HasRawPredictionCol
from xgboost import XGBClassifier, XGBRanker, XGBRegressor
from .core import (  # type: ignore
from .utils import get_class_name
def set_param_attrs(attr_name: str, param: Param) -> None:
    param.typeConverter = param_value_converter
    setattr(estimator, attr_name, param)
    setattr(model, attr_name, param)
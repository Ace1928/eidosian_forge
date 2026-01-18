import logging
import os
from .. import prediction_utils
from .._interfaces import PredictionClient
import numpy as np
from ..prediction_utils import DEFAULT_MODEL_FILE_NAME_JOBLIB
from ..prediction_utils import DEFAULT_MODEL_FILE_NAME_PICKLE
from ..prediction_utils import load_joblib_or_pickle_model
from ..prediction_utils import PredictionError
def create_sk_xg_model(model_path, unused_flags):
    """Create xgboost model or sklearn model from the given model_path.

  Args:
    model_path: path to the directory containing only one of model.joblib or
      model.pkl file. This path can be either a local path or a GCS path.
    unused_flags: Required since model creation for other frameworks needs the
      additional flags params. And model creation is called in a framework
      agnostic manner.

  Returns:
    A xgboost model or sklearn model
  """
    model_obj = load_joblib_or_pickle_model(model_path)
    framework = prediction_utils.detect_sk_xgb_framework_from_obj(model_obj)
    if framework == prediction_utils.SCIKIT_LEARN_FRAMEWORK_NAME:
        return SklearnModel(SklearnClient(model_obj))
    elif framework == prediction_utils.XGBOOST_FRAMEWORK_NAME:
        return XGBoostModel(XgboostClient(model_obj))
    else:
        error_msg = 'Invalid framework detected: {}. Please make sure the model file is supported by either scikit-learn or xgboost.'.format(framework)
        logging.critical(error_msg)
        raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, error_msg)
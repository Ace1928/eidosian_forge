import logging
import os
from .. import prediction_utils
from .._interfaces import PredictionClient
import numpy as np
from ..prediction_utils import DEFAULT_MODEL_FILE_NAME_JOBLIB
from ..prediction_utils import DEFAULT_MODEL_FILE_NAME_PICKLE
from ..prediction_utils import load_joblib_or_pickle_model
from ..prediction_utils import PredictionError
def create_sklearn_client(model_path, **unused_kwargs):
    """Returns a prediction client for the corresponding sklearn model."""
    logging.info('Loading the scikit-learn model file from %s', model_path)
    sklearn_predictor = load_joblib_or_pickle_model(model_path)
    if not sklearn_predictor:
        error_msg = 'Could not find either {} or {} in {}'.format(DEFAULT_MODEL_FILE_NAME_JOBLIB, DEFAULT_MODEL_FILE_NAME_PICKLE, model_path)
        logging.critical(error_msg)
        raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, error_msg)
    if 'sklearn' not in type(sklearn_predictor).__module__:
        error_msg = 'Invalid model type detected: {}.{}. Please make sure the model file is an exported sklearn model or pipeline.'.format(type(sklearn_predictor).__module__, type(sklearn_predictor).__name__)
        logging.critical(error_msg)
        raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, error_msg)
    return SklearnClient(sklearn_predictor)
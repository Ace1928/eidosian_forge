import logging
from bigquery_ml_utils.inference.xgboost_predictor import Predictor
from google.cloud.ml.prediction import copy_model_to_local
from google.cloud.ml.prediction import ENGINE
from google.cloud.ml.prediction import ENGINE_RUN_TIME
from google.cloud.ml.prediction import FRAMEWORK
from google.cloud.ml.prediction import LOCAL_MODEL_PATH
from google.cloud.ml.prediction import PredictionClient
from google.cloud.ml.prediction import PredictionError
from google.cloud.ml.prediction import SESSION_RUN_TIME
from google.cloud.ml.prediction import Stats
from google.cloud.ml.prediction.frameworks.sk_xg_prediction_lib import SklearnModel
def create_xgboost_predictor(model_path, **unused_kwargs):
    """Returns a prediction client for the corresponding xgboost model."""
    logging.info('Downloading the xgboost model from %s to %s', model_path, LOCAL_MODEL_PATH)
    copy_model_to_local(model_path, LOCAL_MODEL_PATH)
    try:
        return Predictor.from_path(LOCAL_MODEL_PATH)
    except Exception as e:
        logging.exception('Exception during loading bqml xgboost model.')
        raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, 'Exception during loading bqml xgboost model: ' + str(e)) from e
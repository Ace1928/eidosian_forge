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
def create_bqml_xgboost_model(model_path, unused_flags):
    """Returns a xgboost model from the given model_path."""
    return BqmlXGBoostModel(BqmlXGBoostClient(create_xgboost_predictor(model_path)))
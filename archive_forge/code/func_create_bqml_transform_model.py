import logging
from bigquery_ml_utils import transform_predictor
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
def create_bqml_transform_model(model_path, unused_flags):
    """Returns a transform model from the given model_path."""
    return BqmlTransformModel(BqmlTransformClient(create_transform_predictor(model_path)))
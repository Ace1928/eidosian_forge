import base64
import collections
import contextlib
import json
import logging
import os
import pickle
import subprocess
import sys
import time
import timeit
from ._interfaces import Model
import six
from tensorflow.python.framework import dtypes  # pylint: disable=g-direct-tensorflow-import
def load_joblib_or_pickle_model(model_path):
    """Loads either a .joblib or .pkl file from GCS or from local.

  Loads one of DEFAULT_MODEL_FILE_NAME_JOBLIB or DEFAULT_MODEL_FILE_NAME_PICKLE
  files if they exist. This is used for both sklearn and xgboost.

  Arguments:
    model_path: The path to the directory that contains the model file. This
      path can be either a local path or a GCS path.

  Raises:
    PredictionError: If there is a problem while loading the file.

  Returns:
    A loaded scikit-learn or xgboost predictor object or None if neither
    DEFAULT_MODEL_FILE_NAME_JOBLIB nor DEFAULT_MODEL_FILE_NAME_PICKLE files are
    found.
  """
    if model_path.startswith('gs://'):
        copy_model_to_local(model_path, LOCAL_MODEL_PATH)
        model_path = LOCAL_MODEL_PATH
    try:
        model_file_name_joblib = os.path.join(model_path, DEFAULT_MODEL_FILE_NAME_JOBLIB)
        model_file_name_pickle = os.path.join(model_path, DEFAULT_MODEL_FILE_NAME_PICKLE)
        if os.path.exists(model_file_name_joblib):
            model_file_name = model_file_name_joblib
            try:
                from sklearn.externals import joblib
            except Exception as e:
                try:
                    import joblib
                except Exception as e:
                    error_msg = 'Could not import joblib module.'
                    logging.exception(error_msg)
                    raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, error_msg)
            try:
                logging.info('Loading model %s using joblib.', model_file_name)
                return joblib.load(model_file_name)
            except KeyError:
                logging.info('Loading model %s using joblib failed. Loading model using xgboost.Booster instead.', model_file_name)
                import xgboost
                booster = xgboost.Booster()
                booster.load_model(model_file_name)
                return booster
        elif os.path.exists(model_file_name_pickle):
            model_file_name = model_file_name_pickle
            logging.info('Loading model %s using pickle.', model_file_name)
            with open(model_file_name, 'rb') as f:
                return pickle.loads(f.read())
        return None
    except Exception as e:
        raw_error_msg = str(e)
        if 'unsupported pickle protocol' in raw_error_msg:
            error_msg = "Could not load the model: {}. {}. Please make sure the model was exported using python {}. Otherwise, please specify the correct 'python_version' parameter when deploying the model.".format(model_file_name, raw_error_msg, sys.version_info[0])
        else:
            error_msg = 'Could not load the model: {}. {}.'.format(model_file_name, raw_error_msg)
        logging.exception(error_msg)
        raise PredictionError(PredictionError.FAILED_TO_LOAD_MODEL, error_msg)
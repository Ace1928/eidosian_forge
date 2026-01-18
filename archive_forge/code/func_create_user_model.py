import inspect
import json
import os
import pydoc  # used for importing python classes from their FQN
import sys
from ._interfaces import Model
from .prediction_utils import PredictionError
def create_user_model(model_path, unused_flags):
    """Loads in the user specified custom Model class.

  Args:
    model_path: The path to either session_bundle or SavedModel.
    unused_flags: Required since model creation for other frameworks needs the
        additional flags params. And model creation is called in a framework
        agnostic manner.

  Returns:
    An instance of a Model.
    Returns None if the user didn't specify the name of the custom
    python class to load in the create_version_request.

  Raises:
    PredictionError: for any of the following:
      (1) the user provided python model class cannot be found
      (2) if the loaded class does not implement the Model interface.
  """
    prediction_class = load_custom_class()
    if not prediction_class:
        return None
    _validate_prediction_class(prediction_class)
    return prediction_class.from_path(model_path)
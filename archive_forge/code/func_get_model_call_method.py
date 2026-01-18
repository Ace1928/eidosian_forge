import os
import keras
import numpy as np
import pandas as pd
from mlflow.exceptions import INVALID_PARAMETER_VALUE, MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.tracking.artifact_utils import _download_artifact_from_uri
from mlflow.utils.annotations import experimental
def get_model_call_method(self):
    if self.save_exported_model:
        return self.model.serve
    else:
        return self.model.predict
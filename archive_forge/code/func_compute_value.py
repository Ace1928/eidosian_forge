import os
import yaml
from mlflow.exceptions import ExecutionException
from mlflow.projects import env_type
from mlflow.tracking import artifact_utils
from mlflow.utils import data_utils
from mlflow.utils.environment import _PYTHON_ENV_FILE_NAME
from mlflow.utils.file_utils import get_local_path_or_none
from mlflow.utils.string_utils import is_string_type, quote
def compute_value(self, param_value, storage_dir, key_position):
    if storage_dir and self.type == 'path':
        return self._compute_path_value(param_value, storage_dir, key_position)
    elif self.type == 'uri':
        return self._compute_uri_value(param_value)
    else:
        return param_value
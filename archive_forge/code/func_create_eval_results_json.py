import json
import os
import tempfile
import time
from datetime import datetime
from typing import List
from mlflow.entities.param import Param
from mlflow.entities.run_status import RunStatus
from mlflow.entities.run_tag import RunTag
from mlflow.utils.file_utils import make_containing_dirs, write_to
from mlflow.utils.mlflow_tags import MLFLOW_LOGGED_ARTIFACTS, MLFLOW_RUN_SOURCE_TYPE
from mlflow.version import VERSION as __version__
def create_eval_results_json(prompt_parameters, model_input, model_output_parameters, model_output):
    columns = [param.key for param in prompt_parameters] + ['prompt', 'output']
    data = [param.value for param in prompt_parameters] + [model_input, model_output]
    updated_columns = columns + [param.key for param in model_output_parameters]
    updated_data = data + [param.value for param in model_output_parameters]
    eval_results = {'columns': updated_columns, 'data': [updated_data]}
    return json.dumps(eval_results)
from typing import Any, List
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils import insecure_hash
def hash_dict_of_arrays(array_dict):
    for key in sorted(array_dict.keys()):
        hash_array(array_dict[key])
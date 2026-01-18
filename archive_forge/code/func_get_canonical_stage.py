from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
def get_canonical_stage(stage):
    key = stage.lower()
    if key not in _CANONICAL_MAPPING:
        raise MlflowException('Invalid Model Version stage: {}. Value must be one of {}.'.format(stage, ', '.join(ALL_STAGES)), INVALID_PARAMETER_VALUE)
    return _CANONICAL_MAPPING[key]
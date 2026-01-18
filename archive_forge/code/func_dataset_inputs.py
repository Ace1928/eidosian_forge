from typing import Any, Dict, List
from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.entities.dataset_input import DatasetInput
from mlflow.protos.service_pb2 import RunInputs as ProtoRunInputs
@property
def dataset_inputs(self) -> List[DatasetInput]:
    """Array of dataset inputs."""
    return self._dataset_inputs
from typing import Optional
from mlflow.entities._mlflow_object import _MlflowObject
from mlflow.protos.service_pb2 import Dataset as ProtoDataset
@property
def source_type(self) -> str:
    """String source_type of the dataset."""
    return self._source_type
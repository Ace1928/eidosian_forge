import warnings
from typing import Any, List, Optional
import entrypoints
from mlflow.data.artifact_dataset_sources import register_artifact_dataset_sources
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
def register_dataset_source(source: DatasetSource):
    """Registers a DatasetSource for use with MLflow Tracking.

    Args:
        source: The DatasetSource to register.
    """
    _dataset_source_registry.register(source)
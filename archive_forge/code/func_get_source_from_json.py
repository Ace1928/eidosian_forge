import warnings
from typing import Any, List, Optional
import entrypoints
from mlflow.data.artifact_dataset_sources import register_artifact_dataset_sources
from mlflow.data.dataset_source import DatasetSource
from mlflow.data.http_dataset_source import HTTPDatasetSource
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import RESOURCE_DOES_NOT_EXIST
def get_source_from_json(self, source_json: str, source_type: str) -> DatasetSource:
    """Parses and returns a DatasetSource object from its JSON representation.

        Args:
            source_json: The JSON representation of the DatasetSource.
            source_type: The string type of the DatasetSource, which indicates how to parse the
                source JSON.
        """
    for source in reversed(self.sources):
        if source._get_source_type() == source_type:
            return source.from_json(source_json)
    raise MlflowException(f'Could not parse dataset source from JSON due to unrecognized source type: {source_type}.', RESOURCE_DOES_NOT_EXIST)
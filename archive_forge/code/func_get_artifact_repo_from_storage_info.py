from typing import List, Optional
from mlflow.entities.model_registry import (
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_uc_registry_messages_pb2 import ModelVersion as ProtoModelVersion
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import (
from mlflow.protos.databricks_uc_registry_messages_pb2 import TemporaryCredentials
from mlflow.store.artifact.artifact_repo import ArtifactRepository
def get_artifact_repo_from_storage_info(storage_location: str, scoped_token: TemporaryCredentials) -> ArtifactRepository:
    """
    Get an ArtifactRepository instance capable of reading/writing to a UC model version's
    file storage location

    Args:
        storage_location: Storage location of the model version
        scoped_token: Protobuf scoped token to use to authenticate to blob storage
    """
    try:
        return _get_artifact_repo_from_storage_info(storage_location=storage_location, scoped_token=scoped_token)
    except ImportError as e:
        raise MlflowException("Unable to import necessary dependencies to access model version files in Unity Catalog. Please ensure you have the necessary dependencies installed, e.g. by running 'pip install mlflow[databricks]' or 'pip install mlflow-skinny[databricks]'") from e
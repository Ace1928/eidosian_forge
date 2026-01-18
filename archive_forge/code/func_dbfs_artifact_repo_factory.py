import json
import os
import posixpath
import mlflow.utils.databricks_utils
from mlflow.entities import FileInfo
from mlflow.environment_variables import MLFLOW_ENABLE_DBFS_FUSE_ARTIFACT_REPO
from mlflow.exceptions import MlflowException
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.databricks_artifact_repo import DatabricksArtifactRepository
from mlflow.store.artifact.local_artifact_repo import LocalArtifactRepository
from mlflow.store.tracking.rest_store import RestStore
from mlflow.tracking._tracking_service import utils
from mlflow.utils.databricks_utils import get_databricks_host_creds
from mlflow.utils.file_utils import relative_path_to_artifact_path
from mlflow.utils.rest_utils import RESOURCE_DOES_NOT_EXIST, http_request, http_request_safe
from mlflow.utils.string_utils import strip_prefix
from mlflow.utils.uri import (
def dbfs_artifact_repo_factory(artifact_uri):
    """
    Returns an ArtifactRepository subclass for storing artifacts on DBFS.

    This factory method is used with URIs of the form ``dbfs:/<path>``. DBFS-backed artifact
    storage can only be used together with the RestStore.

    In the special case where the URI is of the form
    `dbfs:/databricks/mlflow-tracking/<Exp-ID>/<Run-ID>/<path>',
    a DatabricksArtifactRepository is returned. This is capable of storing access controlled
    artifacts.

    Args:
        artifact_uri: DBFS root artifact URI.

    Returns:
        Subclass of ArtifactRepository capable of storing artifacts on DBFS.
    """
    if not is_valid_dbfs_uri(artifact_uri):
        raise MlflowException('DBFS URI must be of the form dbfs:/<path> or ' + 'dbfs://profile@databricks/<path>, but received ' + artifact_uri)
    cleaned_artifact_uri = artifact_uri.rstrip('/')
    db_profile_uri = get_databricks_profile_uri_from_artifact_uri(cleaned_artifact_uri)
    if is_databricks_acled_artifacts_uri(artifact_uri):
        return DatabricksArtifactRepository(cleaned_artifact_uri)
    elif mlflow.utils.databricks_utils.is_dbfs_fuse_available() and MLFLOW_ENABLE_DBFS_FUSE_ARTIFACT_REPO.get() and (not is_databricks_model_registry_artifacts_uri(artifact_uri)) and (db_profile_uri is None or db_profile_uri == 'databricks'):
        final_artifact_uri = remove_databricks_profile_info_from_artifact_uri(cleaned_artifact_uri)
        file_uri = 'file:///dbfs/{}'.format(strip_prefix(final_artifact_uri, 'dbfs:/'))
        return LocalArtifactRepository(file_uri)
    return DbfsRestArtifactRepository(cleaned_artifact_uri)
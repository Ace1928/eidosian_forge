import logging
import os
import urllib.parse
import mlflow
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.store.artifact.databricks_models_artifact_repo import DatabricksModelsArtifactRepository
from mlflow.store.artifact.unity_catalog_models_artifact_repo import (
from mlflow.store.artifact.utils.models import (
from mlflow.utils.file_utils import write_yaml
from mlflow.utils.uri import (
@staticmethod
def is_models_uri(uri):
    return urllib.parse.urlparse(uri).scheme == 'models'
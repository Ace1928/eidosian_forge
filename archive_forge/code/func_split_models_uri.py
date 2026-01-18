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
def split_models_uri(uri):
    """
        Split 'models:/<name>/<version>/path/to/model' into
        ('models:/<name>/<version>', 'path/to/model').
        """
    path = urllib.parse.urlparse(uri).path
    if path.count('/') >= 3 and (not path.endswith('/')):
        splits = path.split('/', 3)
        model_name_and_version = splits[:3]
        artifact_path = splits[-1]
        return ('models:' + '/'.join(model_name_and_version), artifact_path)
    return (uri, '')
import urllib.parse
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.uri import (
@staticmethod
def is_runs_uri(uri):
    return urllib.parse.urlparse(uri).scheme == 'runs'
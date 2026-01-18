import urllib.parse
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.uri import (
@staticmethod
def parse_runs_uri(run_uri):
    parsed = urllib.parse.urlparse(run_uri)
    if parsed.scheme != 'runs':
        raise MlflowException(f'Not a proper runs:/ URI: {run_uri}. ' + "Runs URIs must be of the form 'runs:/<run_id>/run-relative/path/to/artifact'")
    path = parsed.path
    if not path.startswith('/') or len(path) <= 1:
        raise MlflowException(f'Not a proper runs:/ URI: {run_uri}. ' + "Runs URIs must be of the form 'runs:/<run_id>/run-relative/path/to/artifact'")
    path = path[1:]
    path_parts = path.split('/')
    run_id = path_parts[0]
    if run_id == '':
        raise MlflowException(f'Not a proper runs:/ URI: {run_uri}. ' + "Runs URIs must be of the form 'runs:/<run_id>/run-relative/path/to/artifact'")
    artifact_path = '/'.join(path_parts[1:]) if len(path_parts) > 1 else None
    artifact_path = artifact_path if artifact_path != '' else None
    return (run_id, artifact_path)
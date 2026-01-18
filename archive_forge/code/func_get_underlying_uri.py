import urllib.parse
from mlflow.exceptions import MlflowException
from mlflow.store.artifact.artifact_repo import ArtifactRepository
from mlflow.utils.uri import (
@staticmethod
def get_underlying_uri(runs_uri):
    from mlflow.tracking.artifact_utils import get_artifact_uri
    run_id, artifact_path = RunsArtifactRepository.parse_runs_uri(runs_uri)
    tracking_uri = get_databricks_profile_uri_from_artifact_uri(runs_uri)
    uri = get_artifact_uri(run_id, artifact_path, tracking_uri)
    assert not RunsArtifactRepository.is_runs_uri(uri)
    return add_databricks_profile_info_to_artifact_uri(uri, tracking_uri)
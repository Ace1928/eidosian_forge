import urllib.parse
from typing import NamedTuple, Optional
import mlflow.tracking
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import get_databricks_profile_uri_from_artifact_uri, is_databricks_uri
def get_model_name_and_version(client, models_uri):
    model_name, model_version, model_stage, model_alias = _parse_model_uri(models_uri)
    if model_version is not None:
        return (model_name, model_version)
    if model_alias is not None:
        return (model_name, client.get_model_version_by_alias(model_name, model_alias).version)
    return (model_name, str(_get_latest_model_version(client, model_name, model_stage)))
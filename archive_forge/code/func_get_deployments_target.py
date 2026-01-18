import urllib
from typing import Optional
from urllib.parse import urlparse
from mlflow.environment_variables import MLFLOW_DEPLOYMENTS_TARGET
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import append_to_uri_path
def get_deployments_target() -> str:
    """
    Returns the currently set MLflow deployments target iff set.
    If the deployments target has not been set by using ``set_deployments_target``, an
    ``MlflowException`` is raised.
    """
    global _deployments_target
    if _deployments_target is not None:
        return _deployments_target
    elif (uri := MLFLOW_DEPLOYMENTS_TARGET.get()):
        return uri
    else:
        raise MlflowException(f"No deployments target has been set. Please either set the MLflow deployments target via `mlflow.deployments.set_deployments_target()` or set the environment variable {MLFLOW_DEPLOYMENTS_TARGET} to the running deployment server's uri")
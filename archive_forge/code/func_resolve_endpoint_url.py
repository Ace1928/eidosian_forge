import urllib
from typing import Optional
from urllib.parse import urlparse
from mlflow.environment_variables import MLFLOW_DEPLOYMENTS_TARGET
from mlflow.exceptions import MlflowException
from mlflow.utils.uri import append_to_uri_path
def resolve_endpoint_url(base_url: str, endpoint: str) -> str:
    """Performs a validation on whether the returned value is a fully qualified url
    or requires the assembly of a fully qualified url by appending `endpoint`.

    Args:
        base_url: The base URL. Should include the scheme and domain, e.g.,
            ``http://127.0.0.1:6000``.
        endpoint: The endpoint to be appended to the base URL, e.g., ``/api/2.0/endpoints/`` or,
            in the case of Databricks, the fully qualified url.

    Returns:
        The complete URL, either directly returned or formed and returned by joining the
        base URL and the endpoint path.

    """
    return endpoint if _is_valid_uri(endpoint) else append_to_uri_path(base_url, endpoint)
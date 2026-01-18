import base64
import functools
import inspect
import json
import logging
import posixpath
import re
import textwrap
import warnings
from typing import Any, AsyncGenerator, List, Optional
from urllib.parse import urlparse
from starlette.responses import StreamingResponse
from mlflow.environment_variables import MLFLOW_GATEWAY_URI
from mlflow.exceptions import MlflowException
from mlflow.gateway.constants import MLFLOW_AI_GATEWAY_MOSAICML_CHAT_SUPPORTED_MODEL_PREFIXES
from mlflow.utils.uri import append_to_uri_path
def resolve_route_url(base_url: str, route: str) -> str:
    """
    Performs a validation on whether the returned value is a fully qualified url (as the case
    with Databricks) or requires the assembly of a fully qualified url by appending the
    Route return route_url to the base url of the AI Gateway server.

    Args:
        base_url: The base URL. Should include the scheme and domain, e.g.,
            ``http://127.0.0.1:6000``.
        route: The route to be appended to the base URL, e.g., ``/api/2.0/gateway/routes/`` or,
            in the case of Databricks, the fully qualified url.

    Returns:
        The complete URL, either directly returned or formed and returned by joining the
        base URL and the route path.
    """
    return route if _is_valid_uri(route) else append_to_uri_path(base_url, route)
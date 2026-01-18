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
@gateway_deprecated
def get_gateway_uri() -> str:
    """
    Returns the currently set MLflow AI Gateway server uri iff set.
    If the Gateway uri has not been set by using ``set_gateway_uri``, an ``MlflowException``
    is raised.
    """
    global _gateway_uri
    if _gateway_uri is not None:
        return _gateway_uri
    elif (uri := MLFLOW_GATEWAY_URI.get()):
        return uri
    else:
        raise MlflowException(f"No Gateway server uri has been set. Please either set the MLflow Gateway URI via `mlflow.gateway.set_gateway_uri()` or set the environment variable {MLFLOW_GATEWAY_URI} to the running Gateway API server's uri")
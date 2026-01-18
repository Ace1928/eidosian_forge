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
def set_gateway_uri(gateway_uri: str):
    """Sets the uri of a configured and running MLflow AI Gateway server in a global context.
    Providing a valid uri and calling this function is required in order to use the MLflow
    AI Gateway fluent APIs.

    Args:
        gateway_uri: The full uri of a running MLflow AI Gateway server or, if running on
            Databricks, "databricks".
    """
    if not _is_valid_uri(gateway_uri):
        raise MlflowException.invalid_parameter_value('The gateway uri provided is missing required elements. Ensure that the schema and netloc are provided.')
    global _gateway_uri
    _gateway_uri = gateway_uri
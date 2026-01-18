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
def is_valid_endpoint_name(name: str) -> bool:
    """
    Check whether a string contains any URL reserved characters, spaces, or characters other
    than alphanumeric, underscore, hyphen, and dot.

    Returns True if the string doesn't contain any of these characters.
    """
    return bool(re.fullmatch('[\\w\\-\\.]+', name))
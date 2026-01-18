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
def check_configuration_route_name_collisions(config):
    routes = config.get('routes') or config.get('endpoints') or []
    if len(routes) < 2:
        return
    names = [route['name'] for route in routes]
    if len(names) != len(set(names)):
        raise MlflowException.invalid_parameter_value('Duplicate names found in endpoint configurations. Please remove the duplicate endpoint name from the configuration to ensure that endpoints are created properly.')
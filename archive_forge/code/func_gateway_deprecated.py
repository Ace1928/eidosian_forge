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
def gateway_deprecated(obj):
    msg = 'MLflow AI gateway is deprecated and has been replaced by the deployments API for generative AI. See https://mlflow.org/docs/latest/llms/gateway/migration.html for migration.'
    warning = f'\n.. warning::\n\n    {msg}\n'.strip()
    if inspect.isclass(obj):
        original = obj.__init__

        @functools.wraps(original)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, FutureWarning, stacklevel=2)
            return original(*args, **kwargs)
        obj.__init__ = wrapper
        obj.__init__.__doc__ = _prepend(obj.__init__.__doc__, warning)
        return obj
    else:

        @functools.wraps(obj)
        def wrapper(*args, **kwargs):
            warnings.warn(msg, FutureWarning, stacklevel=2)
            return obj(*args, **kwargs)
        wrapper.__doc__ = _prepend(obj.__doc__, warning)
        return wrapper
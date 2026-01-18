import io
import os
import threading
import time
import uuid
from functools import lru_cache
from http import HTTPStatus
from typing import Callable, Tuple, Type, Union
import requests
from requests import Response
from requests.adapters import HTTPAdapter
from requests.models import PreparedRequest
from .. import constants
from . import logging
from ._typing import HTTP_METHOD_T
def reset_sessions() -> None:
    """Reset the cache of sessions.

    Mostly used internally when sessions are reconfigured or an SSLError is raised.
    See [`configure_http_backend`] for more details.
    """
    _get_session_from_cache.cache_clear()
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
def configure_http_backend(backend_factory: BACKEND_FACTORY_T=_default_backend_factory) -> None:
    """
    Configure the HTTP backend by providing a `backend_factory`. Any HTTP calls made by `huggingface_hub` will use a
    Session object instantiated by this factory. This can be useful if you are running your scripts in a specific
    environment requiring custom configuration (e.g. custom proxy or certifications).

    Use [`get_session`] to get a configured Session. Since `requests.Session` is not guaranteed to be thread-safe,
    `huggingface_hub` creates 1 Session instance per thread. They are all instantiated using the same `backend_factory`
    set in [`configure_http_backend`]. A LRU cache is used to cache the created sessions (and connections) between
    calls. Max size is 128 to avoid memory leaks if thousands of threads are spawned.

    See [this issue](https://github.com/psf/requests/issues/2766) to know more about thread-safety in `requests`.

    Example:
    ```py
    import requests
    from huggingface_hub import configure_http_backend, get_session

    # Create a factory function that returns a Session with configured proxies
    def backend_factory() -> requests.Session:
        session = requests.Session()
        session.proxies = {"http": "http://10.10.1.10:3128", "https": "https://10.10.1.11:1080"}
        return session

    # Set it as the default session factory
    configure_http_backend(backend_factory=backend_factory)

    # In practice, this is mostly done internally in `huggingface_hub`
    session = get_session()
    ```
    """
    global _GLOBAL_BACKEND_FACTORY
    _GLOBAL_BACKEND_FACTORY = backend_factory
    reset_sessions()
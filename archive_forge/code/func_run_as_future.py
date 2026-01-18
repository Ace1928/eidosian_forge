from __future__ import annotations
import inspect
import json
import re
import struct
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import wraps
from itertools import islice
from pathlib import Path
from typing import (
from urllib.parse import quote
import requests
from requests.exceptions import HTTPError
from tqdm.auto import tqdm as base_tqdm
from tqdm.contrib.concurrent import thread_map
from ._commit_api import (
from ._inference_endpoints import InferenceEndpoint, InferenceEndpointType
from ._multi_commits import (
from ._space_api import SpaceHardware, SpaceRuntime, SpaceStorage, SpaceVariable
from .community import (
from .constants import (
from .file_download import HfFileMetadata, get_hf_file_metadata, hf_hub_url
from .repocard_data import DatasetCardData, ModelCardData, SpaceCardData
from .utils import (  # noqa: F401 # imported for backward compatibility
from .utils import tqdm as hf_tqdm
from .utils._deprecation import _deprecate_arguments, _deprecate_method
from .utils._typing import CallableT
from .utils.endpoint_helpers import (
def run_as_future(self, fn: Callable[..., R], *args, **kwargs) -> Future[R]:
    """
        Run a method in the background and return a Future instance.

        The main goal is to run methods without blocking the main thread (e.g. to push data during a training).
        Background jobs are queued to preserve order but are not ran in parallel. If you need to speed-up your scripts
        by parallelizing lots of call to the API, you must setup and use your own [ThreadPoolExecutor](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor).

        Note: Most-used methods like [`upload_file`], [`upload_folder`] and [`create_commit`] have a `run_as_future: bool`
        argument to directly call them in the background. This is equivalent to calling `api.run_as_future(...)` on them
        but less verbose.

        Args:
            fn (`Callable`):
                The method to run in the background.
            *args, **kwargs:
                Arguments with which the method will be called.

        Return:
            `Future`: a [Future](https://docs.python.org/3/library/concurrent.futures.html#future-objects) instance to
            get the result of the task.

        Example:
            ```py
            >>> from huggingface_hub import HfApi
            >>> api = HfApi()
            >>> future = api.run_as_future(api.whoami) # instant
            >>> future.done()
            False
            >>> future.result() # wait until complete and return result
            (...)
            >>> future.done()
            True
            ```
        """
    if self._thread_pool is None:
        self._thread_pool = ThreadPoolExecutor(max_workers=1)
    self._thread_pool
    return self._thread_pool.submit(fn, *args, **kwargs)
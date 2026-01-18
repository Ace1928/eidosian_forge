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
@validate_hf_hub_args
def repo_exists(self, repo_id: str, *, repo_type: Optional[str]=None, token: Optional[str]=None) -> bool:
    """
        Checks if a repository exists on the Hugging Face Hub.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if getting repository info from a dataset or a space,
                `None` or `"model"` if getting repository info from a model. Default is `None`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            True if the repository exists, False otherwise.

        Examples:
            ```py
            >>> from huggingface_hub import repo_exists
            >>> repo_exists("google/gemma-7b")
            True
            >>> repo_exists("google/not-a-repo")
            False
            ```
        """
    try:
        self.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
        return True
    except GatedRepoError:
        return True
    except RepositoryNotFoundError:
        return False
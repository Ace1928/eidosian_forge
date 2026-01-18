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
def like(self, repo_id: str, *, token: Optional[str]=None, repo_type: Optional[str]=None) -> None:
    """
        Like a given repo on the Hub (e.g. set as favorite).

        See also [`unlike`] and [`list_liked_repos`].

        Args:
            repo_id (`str`):
                The repository to like. Example: `"user/my-cool-model"`.

            token (`str`, *optional*):
                Authentication token. Will default to the stored token.

            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if liking a dataset or space, `None` or
                `"model"` if liking a model. Default is `None`.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private
                but not authenticated or repo does not exist.

        Example:
        ```python
        >>> from huggingface_hub import like, list_liked_repos, unlike
        >>> like("gpt2")
        >>> "gpt2" in list_liked_repos().models
        True
        >>> unlike("gpt2")
        >>> "gpt2" in list_liked_repos().models
        False
        ```
        """
    if repo_type is None:
        repo_type = REPO_TYPE_MODEL
    response = get_session().post(url=f'{self.endpoint}/api/{repo_type}s/{repo_id}/like', headers=self._build_hf_headers(token=token))
    hf_raise_for_status(response)
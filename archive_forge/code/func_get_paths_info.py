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
def get_paths_info(self, repo_id: str, paths: Union[List[str], str], *, expand: bool=False, revision: Optional[str]=None, repo_type: Optional[str]=None, token: Optional[Union[bool, str]]=None) -> List[Union[RepoFile, RepoFolder]]:
    """
        Get information about a repo's paths.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated by a `/`.
            paths (`Union[List[str], str]`, *optional*):
                The paths to get information about. If a path do not exist, it is ignored without raising
                an exception.
            expand (`bool`, *optional*, defaults to `False`):
                Whether to fetch more information about the paths (e.g. last commit and files' security scan results). This
                operation is more expensive for the server so only 50 results are returned per page (instead of 1000).
                As pagination is implemented in `huggingface_hub`, this is transparent for you except for the time it
                takes to get the results.
            revision (`str`, *optional*):
                The revision of the repository from which to get the information. Defaults to `"main"` branch.
            repo_type (`str`, *optional*):
                The type of the repository from which to get the information (`"model"`, `"dataset"` or `"space"`.
                Defaults to `"model"`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token). If `None` or `True` and
                machine is logged in (through `huggingface-cli login` or [`~huggingface_hub.login`]), token will be
                retrieved from the cache. If `False`, token is not sent in the request header.

        Returns:
            `List[Union[RepoFile, RepoFolder]]`:
                The information about the paths, as a list of [`RepoFile`] and [`RepoFolder`] objects.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private but not authenticated or repo
                does not exist.
            [`~utils.RevisionNotFoundError`]:
                If revision is not found (error 404) on the repo.

        Example:
        ```py
        >>> from huggingface_hub import get_paths_info
        >>> paths_info = get_paths_info("allenai/c4", ["README.md", "en"], repo_type="dataset")
        >>> paths_info
        [
            RepoFile(path='README.md', size=2379, blob_id='f84cb4c97182890fc1dbdeaf1a6a468fd27b4fff', lfs=None, last_commit=None, security=None),
            RepoFolder(path='en', tree_id='dc943c4c40f53d02b31ced1defa7e5f438d5862e', last_commit=None)
        ]
        ```
        """
    repo_type = repo_type or REPO_TYPE_MODEL
    revision = quote(revision, safe='') if revision is not None else DEFAULT_REVISION
    headers = self._build_hf_headers(token=token)
    response = get_session().post(f'{self.endpoint}/api/{repo_type}s/{repo_id}/paths-info/{revision}', data={'paths': paths if isinstance(paths, list) else [paths], 'expand': expand}, headers=headers)
    hf_raise_for_status(response)
    paths_info = response.json()
    return [RepoFile(**path_info) if path_info['type'] == 'file' else RepoFolder(**path_info) for path_info in paths_info]
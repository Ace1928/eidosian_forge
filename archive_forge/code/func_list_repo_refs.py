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
def list_repo_refs(self, repo_id: str, *, repo_type: Optional[str]=None, include_pull_requests: bool=False, token: Optional[Union[bool, str]]=None) -> GitRefs:
    """
        Get the list of refs of a given repo (both tags and branches).

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if listing refs from a dataset or a Space,
                `None` or `"model"` if listing from a model. Default is `None`.
            include_pull_requests (`bool`, *optional*):
                Whether to include refs from pull requests in the list. Defaults to `False`.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Example:
        ```py
        >>> from huggingface_hub import HfApi
        >>> api = HfApi()
        >>> api.list_repo_refs("gpt2")
        GitRefs(branches=[GitRefInfo(name='main', ref='refs/heads/main', target_commit='e7da7f221d5bf496a48136c0cd264e630fe9fcc8')], converts=[], tags=[])

        >>> api.list_repo_refs("bigcode/the-stack", repo_type='dataset')
        GitRefs(
            branches=[
                GitRefInfo(name='main', ref='refs/heads/main', target_commit='18edc1591d9ce72aa82f56c4431b3c969b210ae3'),
                GitRefInfo(name='v1.1.a1', ref='refs/heads/v1.1.a1', target_commit='f9826b862d1567f3822d3d25649b0d6d22ace714')
            ],
            converts=[],
            tags=[
                GitRefInfo(name='v1.0', ref='refs/tags/v1.0', target_commit='c37a8cd1e382064d8aced5e05543c5f7753834da')
            ]
        )
        ```

        Returns:
            [`GitRefs`]: object containing all information about branches and tags for a
            repo on the Hub.
        """
    repo_type = repo_type or REPO_TYPE_MODEL
    response = get_session().get(f'{self.endpoint}/api/{repo_type}s/{repo_id}/refs', headers=self._build_hf_headers(token=token), params={'include_prs': 1} if include_pull_requests else {})
    hf_raise_for_status(response)
    data = response.json()

    def _format_as_git_ref_info(item: Dict) -> GitRefInfo:
        return GitRefInfo(name=item['name'], ref=item['ref'], target_commit=item['targetCommit'])
    return GitRefs(branches=[_format_as_git_ref_info(item) for item in data['branches']], converts=[_format_as_git_ref_info(item) for item in data['converts']], tags=[_format_as_git_ref_info(item) for item in data['tags']], pull_requests=[_format_as_git_ref_info(item) for item in data['pullRequests']] if include_pull_requests else None)
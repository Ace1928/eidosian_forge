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
def get_discussion_details(self, repo_id: str, discussion_num: int, *, repo_type: Optional[str]=None, token: Optional[str]=None) -> DiscussionWithDetails:
    """Fetches a Discussion's / Pull Request 's details from the Hub.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            discussion_num (`int`):
                The number of the Discussion or Pull Request . Must be a strictly positive integer.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        Returns: [`DiscussionWithDetails`]

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
    if not isinstance(discussion_num, int) or discussion_num <= 0:
        raise ValueError('Invalid discussion_num, must be a positive integer')
    if repo_type not in REPO_TYPES:
        raise ValueError(f'Invalid repo type, must be one of {REPO_TYPES}')
    if repo_type is None:
        repo_type = REPO_TYPE_MODEL
    path = f'{self.endpoint}/api/{repo_type}s/{repo_id}/discussions/{discussion_num}'
    headers = self._build_hf_headers(token=token)
    resp = get_session().get(path, params={'diff': '1'}, headers=headers)
    hf_raise_for_status(resp)
    discussion_details = resp.json()
    is_pull_request = discussion_details['isPullRequest']
    target_branch = discussion_details['changes']['base'] if is_pull_request else None
    conflicting_files = discussion_details['filesWithConflicts'] if is_pull_request else None
    merge_commit_oid = discussion_details['changes'].get('mergeCommitId', None) if is_pull_request else None
    return DiscussionWithDetails(title=discussion_details['title'], num=discussion_details['num'], author=discussion_details.get('author', {}).get('name', 'deleted'), created_at=parse_datetime(discussion_details['createdAt']), status=discussion_details['status'], repo_id=discussion_details['repo']['name'], repo_type=discussion_details['repo']['type'], is_pull_request=discussion_details['isPullRequest'], events=[deserialize_event(evt) for evt in discussion_details['events']], conflicting_files=conflicting_files, target_branch=target_branch, merge_commit_oid=merge_commit_oid, diff=discussion_details.get('diff'), endpoint=self.endpoint)
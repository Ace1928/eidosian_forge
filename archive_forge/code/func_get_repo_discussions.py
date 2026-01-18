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
def get_repo_discussions(self, repo_id: str, *, author: Optional[str]=None, discussion_type: Optional[DiscussionTypeFilter]=None, discussion_status: Optional[DiscussionStatusFilter]=None, repo_type: Optional[str]=None, token: Optional[str]=None) -> Iterator[Discussion]:
    """
        Fetches Discussions and Pull Requests for the given repo.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            author (`str`, *optional*):
                Pass a value to filter by discussion author. `None` means no filter.
                Default is `None`.
            discussion_type (`str`, *optional*):
                Set to `"pull_request"` to fetch only pull requests, `"discussion"`
                to fetch only discussions. Set to `"all"` or `None` to fetch both.
                Default is `None`.
            discussion_status (`str`, *optional*):
                Set to `"open"` (respectively `"closed"`) to fetch only open
                (respectively closed) discussions. Set to `"all"` or `None`
                to fetch both.
                Default is `None`.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if fetching from a dataset or
                space, `None` or `"model"` if fetching from a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token).

        Returns:
            `Iterator[Discussion]`: An iterator of [`Discussion`] objects.

        Example:
            Collecting all discussions of a repo in a list:

            ```python
            >>> from huggingface_hub import get_repo_discussions
            >>> discussions_list = list(get_repo_discussions(repo_id="bert-base-uncased"))
            ```

            Iterating over discussions of a repo:

            ```python
            >>> from huggingface_hub import get_repo_discussions
            >>> for discussion in get_repo_discussions(repo_id="bert-base-uncased"):
            ...     print(discussion.num, discussion.title)
            ```
        """
    if repo_type not in REPO_TYPES:
        raise ValueError(f'Invalid repo type, must be one of {REPO_TYPES}')
    if repo_type is None:
        repo_type = REPO_TYPE_MODEL
    if discussion_type is not None and discussion_type not in DISCUSSION_TYPES:
        raise ValueError(f'Invalid discussion_type, must be one of {DISCUSSION_TYPES}')
    if discussion_status is not None and discussion_status not in DISCUSSION_STATUS:
        raise ValueError(f'Invalid discussion_status, must be one of {DISCUSSION_STATUS}')
    headers = self._build_hf_headers(token=token)
    path = f'{self.endpoint}/api/{repo_type}s/{repo_id}/discussions'
    params: Dict[str, Union[str, int]] = {}
    if discussion_type is not None:
        params['type'] = discussion_type
    if discussion_status is not None:
        params['status'] = discussion_status
    if author is not None:
        params['author'] = author

    def _fetch_discussion_page(page_index: int):
        params['p'] = page_index
        resp = get_session().get(path, headers=headers, params=params)
        hf_raise_for_status(resp)
        paginated_discussions = resp.json()
        total = paginated_discussions['count']
        start = paginated_discussions['start']
        discussions = paginated_discussions['discussions']
        has_next = start + len(discussions) < total
        return (discussions, has_next)
    has_next, page_index = (True, 0)
    while has_next:
        discussions, has_next = _fetch_discussion_page(page_index=page_index)
        for discussion in discussions:
            yield Discussion(title=discussion['title'], num=discussion['num'], author=discussion.get('author', {}).get('name', 'deleted'), created_at=parse_datetime(discussion['createdAt']), status=discussion['status'], repo_id=discussion['repo']['name'], repo_type=discussion['repo']['type'], is_pull_request=discussion['isPullRequest'], endpoint=self.endpoint)
        page_index = page_index + 1
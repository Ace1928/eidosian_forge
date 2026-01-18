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
def super_squash_history(self, repo_id: str, *, branch: Optional[str]=None, commit_message: Optional[str]=None, repo_type: Optional[str]=None, token: Optional[str]=None) -> None:
    """Squash commit history on a branch for a repo on the Hub.

        Squashing the repo history is useful when you know you'll make hundreds of commits and you don't want to
        clutter the history. Squashing commits can only be performed from the head of a branch.

        <Tip warning={true}>

        Once squashed, the commit history cannot be retrieved. This is a non-revertible operation.

        </Tip>

        <Tip warning={true}>

        Once the history of a branch has been squashed, it is not possible to merge it back into another branch since
        their history will have diverged.

        </Tip>

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated by a `/`.
            branch (`str`, *optional*):
                The branch to squash. Defaults to the head of the `"main"` branch.
            commit_message (`str`, *optional*):
                The commit message to use for the squashed commit.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if listing commits from a dataset or a Space, `None` or `"model"` if
                listing from a model. Default is `None`.
            token (`str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token). If the machine is logged in
                (through `huggingface-cli login` or [`~huggingface_hub.login`]), token can be automatically retrieved
                from the cache.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If repository is not found (error 404): wrong repo_id/repo_type, private but not authenticated or repo
                does not exist.
            [`~utils.RevisionNotFoundError`]:
                If the branch to squash cannot be found.
            [`~utils.BadRequestError`]:
                If invalid reference for a branch. You cannot squash history on tags.

        Example:
        ```py
        >>> from huggingface_hub import HfApi
        >>> api = HfApi()

        # Create repo
        >>> repo_id = api.create_repo("test-squash").repo_id

        # Make a lot of commits.
        >>> api.upload_file(repo_id=repo_id, path_in_repo="file.txt", path_or_fileobj=b"content")
        >>> api.upload_file(repo_id=repo_id, path_in_repo="lfs.bin", path_or_fileobj=b"content")
        >>> api.upload_file(repo_id=repo_id, path_in_repo="file.txt", path_or_fileobj=b"another_content")

        # Squash history
        >>> api.super_squash_history(repo_id=repo_id)
        ```
        """
    if repo_type is None:
        repo_type = REPO_TYPE_MODEL
    if repo_type not in REPO_TYPES:
        raise ValueError('Invalid repo type')
    if branch is None:
        branch = DEFAULT_REVISION
    url = f'{self.endpoint}/api/{repo_type}s/{repo_id}/super-squash/{branch}'
    headers = self._build_hf_headers(token=token, is_write_action=True)
    commit_message = commit_message or f"Super-squash branch '{branch}' using huggingface_hub"
    response = get_session().post(url=url, headers=headers, json={'message': commit_message})
    hf_raise_for_status(response)
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
def delete_repo(self, repo_id: str, *, token: Optional[str]=None, repo_type: Optional[str]=None, missing_ok: bool=False) -> None:
    """
        Delete a repo from the HuggingFace Hub. CAUTION: this is irreversible.

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model.
            missing_ok (`bool`, *optional*, defaults to `False`):
                If `True`, do not raise an error if repo does not exist.

        Raises:
            - [`~utils.RepositoryNotFoundError`]
              If the repository to delete from cannot be found and `missing_ok` is set to False (default).
        """
    organization, name = repo_id.split('/') if '/' in repo_id else (None, repo_id)
    path = f'{self.endpoint}/api/repos/delete'
    if repo_type not in REPO_TYPES:
        raise ValueError('Invalid repo type')
    json = {'name': name, 'organization': organization}
    if repo_type is not None:
        json['type'] = repo_type
    headers = self._build_hf_headers(token=token, is_write_action=True)
    r = get_session().delete(path, headers=headers, json=json)
    try:
        hf_raise_for_status(r)
    except RepositoryNotFoundError:
        if not missing_ok:
            raise
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
def move_repo(self, from_id: str, to_id: str, *, repo_type: Optional[str]=None, token: Optional[str]=None):
    """
        Moving a repository from namespace1/repo_name1 to namespace2/repo_name2

        Note there are certain limitations. For more information about moving
        repositories, please see
        https://hf.co/docs/hub/repositories-settings#renaming-or-transferring-a-repo.

        Args:
            from_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`. Original repository identifier.
            to_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`. Final repository identifier.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)

        <Tip>

        Raises the following errors:

            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>
        """
    if len(from_id.split('/')) != 2:
        raise ValueError(f'Invalid repo_id: {from_id}. It should have a namespace (:namespace:/:repo_name:)')
    if len(to_id.split('/')) != 2:
        raise ValueError(f'Invalid repo_id: {to_id}. It should have a namespace (:namespace:/:repo_name:)')
    if repo_type is None:
        repo_type = REPO_TYPE_MODEL
    json = {'fromRepo': from_id, 'toRepo': to_id, 'type': repo_type}
    path = f'{self.endpoint}/api/repos/move'
    headers = self._build_hf_headers(token=token, is_write_action=True)
    r = get_session().post(path, headers=headers, json=json)
    try:
        hf_raise_for_status(r)
    except HfHubHTTPError as e:
        e.append_to_message('\nFor additional documentation please see https://hf.co/docs/hub/repositories-settings#renaming-or-transferring-a-repo.')
        raise
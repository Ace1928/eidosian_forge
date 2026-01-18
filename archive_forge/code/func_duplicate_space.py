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
def duplicate_space(self, from_id: str, to_id: Optional[str]=None, *, private: Optional[bool]=None, token: Optional[str]=None, exist_ok: bool=False, hardware: Optional[SpaceHardware]=None, storage: Optional[SpaceStorage]=None, sleep_time: Optional[int]=None, secrets: Optional[List[Dict[str, str]]]=None, variables: Optional[List[Dict[str, str]]]=None) -> RepoUrl:
    """Duplicate a Space.

        Programmatically duplicate a Space. The new Space will be created in your account and will be in the same state
        as the original Space (running or paused). You can duplicate a Space no matter the current state of a Space.

        Args:
            from_id (`str`):
                ID of the Space to duplicate. Example: `"pharma/CLIP-Interrogator"`.
            to_id (`str`, *optional*):
                ID of the new Space. Example: `"dog/CLIP-Interrogator"`. If not provided, the new Space will have the same
                name as the original Space, but in your account.
            private (`bool`, *optional*):
                Whether the new Space should be private or not. Defaults to the same privacy as the original Space.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if not provided.
            exist_ok (`bool`, *optional*, defaults to `False`):
                If `True`, do not raise an error if repo already exists.
            hardware (`SpaceHardware` or `str`, *optional*):
                Choice of Hardware. Example: `"t4-medium"`. See [`SpaceHardware`] for a complete list.
            storage (`SpaceStorage` or `str`, *optional*):
                Choice of persistent storage tier. Example: `"small"`. See [`SpaceStorage`] for a complete list.
            sleep_time (`int`, *optional*):
                Number of seconds of inactivity to wait before a Space is put to sleep. Set to `-1` if you don't want
                your Space to sleep (default behavior for upgraded hardware). For free hardware, you can't configure
                the sleep time (value is fixed to 48 hours of inactivity).
                See https://huggingface.co/docs/hub/spaces-gpus#sleep-time for more details.
            secrets (`List[Dict[str, str]]`, *optional*):
                A list of secret keys to set in your Space. Each item is in the form `{"key": ..., "value": ..., "description": ...}` where description is optional.
                For more details, see https://huggingface.co/docs/hub/spaces-overview#managing-secrets.
            variables (`List[Dict[str, str]]`, *optional*):
                A list of public environment variables to set in your Space. Each item is in the form `{"key": ..., "value": ..., "description": ...}` where description is optional.
                For more details, see https://huggingface.co/docs/hub/spaces-overview#managing-secrets-and-environment-variables.

        Returns:
            [`RepoUrl`]: URL to the newly created repo. Value is a subclass of `str` containing
            attributes like `endpoint`, `repo_type` and `repo_id`.

        Raises:
            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`~utils.RepositoryNotFoundError`]
              If one of `from_id` or `to_id` cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        Example:
        ```python
        >>> from huggingface_hub import duplicate_space

        # Duplicate a Space to your account
        >>> duplicate_space("multimodalart/dreambooth-training")
        RepoUrl('https://huggingface.co/spaces/nateraw/dreambooth-training',...)

        # Can set custom destination id and visibility flag.
        >>> duplicate_space("multimodalart/dreambooth-training", to_id="my-dreambooth", private=True)
        RepoUrl('https://huggingface.co/spaces/nateraw/my-dreambooth',...)
        ```
        """
    parsed_to_id = RepoUrl(to_id) if to_id is not None else None
    to_namespace = parsed_to_id.namespace if parsed_to_id is not None and parsed_to_id.namespace is not None else self.whoami(token)['name']
    to_repo_name = parsed_to_id.repo_name if to_id is not None else RepoUrl(from_id).repo_name
    payload: Dict[str, Any] = {'repository': f'{to_namespace}/{to_repo_name}'}
    keys = ['private', 'hardware', 'storageTier', 'sleepTimeSeconds', 'secrets', 'variables']
    values = [private, hardware, storage, sleep_time, secrets, variables]
    payload.update({k: v for k, v in zip(keys, values) if v is not None})
    if sleep_time is not None and hardware == SpaceHardware.CPU_BASIC:
        warnings.warn("If your Space runs on the default 'cpu-basic' hardware, it will go to sleep if inactive for more than 48 hours. This value is not configurable. If you don't want your Space to deactivate or if you want to set a custom sleep time, you need to upgrade to a paid Hardware.", UserWarning)
    r = get_session().post(f'{self.endpoint}/api/spaces/{from_id}/duplicate', headers=self._build_hf_headers(token=token, is_write_action=True), json=payload)
    try:
        hf_raise_for_status(r)
    except HTTPError as err:
        if exist_ok and err.response.status_code == 409:
            pass
        else:
            raise
    return RepoUrl(r.json()['url'], endpoint=self.endpoint)
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
def create_collection(self, title: str, *, namespace: Optional[str]=None, description: Optional[str]=None, private: bool=False, exists_ok: bool=False, token: Optional[str]=None) -> Collection:
    """Create a new Collection on the Hub.

        Args:
            title (`str`):
                Title of the collection to create. Example: `"Recent models"`.
            namespace (`str`, *optional*):
                Namespace of the collection to create (username or org). Will default to the owner name.
            description (`str`, *optional*):
                Description of the collection to create.
            private (`bool`, *optional*):
                Whether the collection should be private or not. Defaults to `False` (i.e. public collection).
            exists_ok (`bool`, *optional*):
                If `True`, do not raise an error if collection already exists.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if not provided.

        Returns: [`Collection`]

        Example:

        ```py
        >>> from huggingface_hub import create_collection
        >>> collection = create_collection(
        ...     title="ICCV 2023",
        ...     description="Portfolio of models, papers and demos I presented at ICCV 2023",
        ... )
        >>> collection.slug
        "username/iccv-2023-64f9a55bb3115b4f513ec026"
        ```
        """
    if namespace is None:
        namespace = self.whoami(token)['name']
    payload = {'title': title, 'namespace': namespace, 'private': private}
    if description is not None:
        payload['description'] = description
    r = get_session().post(f'{self.endpoint}/api/collections', headers=self._build_hf_headers(token=token), json=payload)
    try:
        hf_raise_for_status(r)
    except HTTPError as err:
        if exists_ok and err.response.status_code == 409:
            slug = r.json()['slug']
            return self.get_collection(slug, token=token)
        else:
            raise
    return Collection(**{**r.json(), 'endpoint': self.endpoint})
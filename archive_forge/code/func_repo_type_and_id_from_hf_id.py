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
def repo_type_and_id_from_hf_id(hf_id: str, hub_url: Optional[str]=None) -> Tuple[Optional[str], Optional[str], str]:
    """
    Returns the repo type and ID from a huggingface.co URL linking to a
    repository

    Args:
        hf_id (`str`):
            An URL or ID of a repository on the HF hub. Accepted values are:

            - https://huggingface.co/<repo_type>/<namespace>/<repo_id>
            - https://huggingface.co/<namespace>/<repo_id>
            - hf://<repo_type>/<namespace>/<repo_id>
            - hf://<namespace>/<repo_id>
            - <repo_type>/<namespace>/<repo_id>
            - <namespace>/<repo_id>
            - <repo_id>
        hub_url (`str`, *optional*):
            The URL of the HuggingFace Hub, defaults to https://huggingface.co

    Returns:
        A tuple with three items: repo_type (`str` or `None`), namespace (`str` or
        `None`) and repo_id (`str`).

    Raises:
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If URL cannot be parsed.
        - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
            If `repo_type` is unknown.
    """
    input_hf_id = hf_id
    hub_url = re.sub('https?://', '', hub_url if hub_url is not None else ENDPOINT)
    is_hf_url = hub_url in hf_id and '@' not in hf_id
    HFFS_PREFIX = 'hf://'
    if hf_id.startswith(HFFS_PREFIX):
        hf_id = hf_id[len(HFFS_PREFIX):]
    url_segments = hf_id.split('/')
    is_hf_id = len(url_segments) <= 3
    namespace: Optional[str]
    if is_hf_url:
        namespace, repo_id = url_segments[-2:]
        if namespace == hub_url:
            namespace = None
        if len(url_segments) > 2 and hub_url not in url_segments[-3]:
            repo_type = url_segments[-3]
        elif namespace in REPO_TYPES_MAPPING:
            repo_type = REPO_TYPES_MAPPING[namespace]
            namespace = None
        else:
            repo_type = None
    elif is_hf_id:
        if len(url_segments) == 3:
            repo_type, namespace, repo_id = url_segments[-3:]
        elif len(url_segments) == 2:
            if url_segments[0] in REPO_TYPES_MAPPING:
                repo_type = REPO_TYPES_MAPPING[url_segments[0]]
                namespace = None
                repo_id = hf_id.split('/')[-1]
            else:
                namespace, repo_id = hf_id.split('/')[-2:]
                repo_type = None
        else:
            repo_id = url_segments[0]
            namespace, repo_type = (None, None)
    else:
        raise ValueError(f'Unable to retrieve user and repo ID from the passed HF ID: {hf_id}')
    if repo_type in REPO_TYPES_MAPPING:
        repo_type = REPO_TYPES_MAPPING[repo_type]
    if repo_type == '':
        repo_type = None
    if repo_type not in REPO_TYPES:
        raise ValueError(f"Unknown `repo_type`: '{repo_type}' ('{input_hf_id}')")
    return (repo_type, namespace, repo_id)
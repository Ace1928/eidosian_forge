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
def get_full_repo_name(self, model_id: str, *, organization: Optional[str]=None, token: Optional[Union[bool, str]]=None):
    """
        Returns the repository name for a given model ID and optional
        organization.

        Args:
            model_id (`str`):
                The name of the model.
            organization (`str`, *optional*):
                If passed, the repository name will be in the organization
                namespace instead of the user namespace.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                If `None` or `True` and machine is logged in (through `huggingface-cli login`
                or [`~huggingface_hub.login`]), token will be retrieved from the cache.
                If `False`, token is not sent in the request header.

        Returns:
            `str`: The repository name in the user's namespace
            ({username}/{model_id}) if no organization is passed, and under the
            organization namespace ({organization}/{model_id}) otherwise.
        """
    if organization is None:
        if '/' in model_id:
            username = model_id.split('/')[0]
        else:
            username = self.whoami(token=token)['name']
        return f'{username}/{model_id}'
    else:
        return f'{organization}/{model_id}'
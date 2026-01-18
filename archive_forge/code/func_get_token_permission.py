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
def get_token_permission(self, token: Optional[str]=None) -> Literal['read', 'write', None]:
    """
        Check if a given `token` is valid and return its permissions.

        For more details about tokens, please refer to https://huggingface.co/docs/hub/security-tokens#what-are-user-access-tokens.

        Args:
            token (`str`, *optional*):
                The token to check for validity. Defaults to the one saved locally.

        Returns:
            `Literal["read", "write", None]`: Permission granted by the token ("read" or "write"). Returns `None` if no
            token passed or token is invalid.
        """
    try:
        return self.whoami(token=token)['auth']['accessToken']['role']
    except (LocalTokenNotFoundError, HTTPError):
        return None
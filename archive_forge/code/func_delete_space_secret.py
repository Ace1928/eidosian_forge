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
def delete_space_secret(self, repo_id: str, key: str, *, token: Optional[str]=None) -> None:
    """Deletes a secret from a Space.

        Secrets allow to set secret keys or tokens to a Space without hardcoding them.
        For more details, see https://huggingface.co/docs/hub/spaces-overview#managing-secrets.

        Args:
            repo_id (`str`):
                ID of the repo to update. Example: `"bigcode/in-the-stack"`.
            key (`str`):
                Secret key. Example: `"GITHUB_API_KEY"`.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if not provided.
        """
    r = get_session().delete(f'{self.endpoint}/api/spaces/{repo_id}/secrets', headers=self._build_hf_headers(token=token), json={'key': key})
    hf_raise_for_status(r)
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
def request_space_storage(self, repo_id: str, storage: SpaceStorage, *, token: Optional[str]=None) -> SpaceRuntime:
    """Request persistent storage for a Space.

        Args:
            repo_id (`str`):
                ID of the Space to update. Example: `"HuggingFaceH4/open_llm_leaderboard"`.
            storage (`str` or [`SpaceStorage`]):
               Storage tier. Either 'small', 'medium', or 'large'.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if not provided.
        Returns:
            [`SpaceRuntime`]: Runtime information about a Space including Space stage and hardware.

        <Tip>

        It is not possible to decrease persistent storage after its granted. To do so, you must delete it
        via [`delete_space_storage`].

        </Tip>
        """
    payload: Dict[str, SpaceStorage] = {'tier': storage}
    r = get_session().post(f'{self.endpoint}/api/spaces/{repo_id}/storage', headers=self._build_hf_headers(token=token), json=payload)
    hf_raise_for_status(r)
    return SpaceRuntime(r.json())
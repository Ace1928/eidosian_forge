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
def request_space_hardware(self, repo_id: str, hardware: SpaceHardware, *, token: Optional[str]=None, sleep_time: Optional[int]=None) -> SpaceRuntime:
    """Request new hardware for a Space.

        Args:
            repo_id (`str`):
                ID of the repo to update. Example: `"bigcode/in-the-stack"`.
            hardware (`str` or [`SpaceHardware`]):
                Hardware on which to run the Space. Example: `"t4-medium"`.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if not provided.
            sleep_time (`int`, *optional*):
                Number of seconds of inactivity to wait before a Space is put to sleep. Set to `-1` if you don't want
                your Space to sleep (default behavior for upgraded hardware). For free hardware, you can't configure
                the sleep time (value is fixed to 48 hours of inactivity).
                See https://huggingface.co/docs/hub/spaces-gpus#sleep-time for more details.
        Returns:
            [`SpaceRuntime`]: Runtime information about a Space including Space stage and hardware.

        <Tip>

        It is also possible to request hardware directly when creating the Space repo! See [`create_repo`] for details.

        </Tip>
        """
    if sleep_time is not None and hardware == SpaceHardware.CPU_BASIC:
        warnings.warn("If your Space runs on the default 'cpu-basic' hardware, it will go to sleep if inactive for more than 48 hours. This value is not configurable. If you don't want your Space to deactivate or if you want to set a custom sleep time, you need to upgrade to a paid Hardware.", UserWarning)
    payload: Dict[str, Any] = {'flavor': hardware}
    if sleep_time is not None:
        payload['sleepTimeSeconds'] = sleep_time
    r = get_session().post(f'{self.endpoint}/api/spaces/{repo_id}/hardware', headers=self._build_hf_headers(token=token), json=payload)
    hf_raise_for_status(r)
    return SpaceRuntime(r.json())
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
def set_space_sleep_time(self, repo_id: str, sleep_time: int, *, token: Optional[str]=None) -> SpaceRuntime:
    """Set a custom sleep time for a Space running on upgraded hardware..

        Your Space will go to sleep after X seconds of inactivity. You are not billed when your Space is in "sleep"
        mode. If a new visitor lands on your Space, it will "wake it up". Only upgraded hardware can have a
        configurable sleep time. To know more about the sleep stage, please refer to
        https://huggingface.co/docs/hub/spaces-gpus#sleep-time.

        Args:
            repo_id (`str`):
                ID of the repo to update. Example: `"bigcode/in-the-stack"`.
            sleep_time (`int`, *optional*):
                Number of seconds of inactivity to wait before a Space is put to sleep. Set to `-1` if you don't want
                your Space to pause (default behavior for upgraded hardware). For free hardware, you can't configure
                the sleep time (value is fixed to 48 hours of inactivity).
                See https://huggingface.co/docs/hub/spaces-gpus#sleep-time for more details.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if not provided.
        Returns:
            [`SpaceRuntime`]: Runtime information about a Space including Space stage and hardware.

        <Tip>

        It is also possible to set a custom sleep time when requesting hardware with [`request_space_hardware`].

        </Tip>
        """
    r = get_session().post(f'{self.endpoint}/api/spaces/{repo_id}/sleeptime', headers=self._build_hf_headers(token=token), json={'seconds': sleep_time})
    hf_raise_for_status(r)
    runtime = SpaceRuntime(r.json())
    hardware = runtime.requested_hardware or runtime.hardware
    if hardware == SpaceHardware.CPU_BASIC:
        warnings.warn("If your Space runs on the default 'cpu-basic' hardware, it will go to sleep if inactive for more than 48 hours. This value is not configurable. If you don't want your Space to deactivate or if you want to set a custom sleep time, you need to upgrade to a paid Hardware.", UserWarning)
    return runtime
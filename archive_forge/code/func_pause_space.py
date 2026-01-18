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
def pause_space(self, repo_id: str, *, token: Optional[str]=None) -> SpaceRuntime:
    """Pause your Space.

        A paused Space stops executing until manually restarted by its owner. This is different from the sleeping
        state in which free Spaces go after 48h of inactivity. Paused time is not billed to your account, no matter the
        hardware you've selected. To restart your Space, use [`restart_space`] and go to your Space settings page.

        For more details, please visit [the docs](https://huggingface.co/docs/hub/spaces-gpus#pause).

        Args:
            repo_id (`str`):
                ID of the Space to pause. Example: `"Salesforce/BLIP2"`.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if not provided.

        Returns:
            [`SpaceRuntime`]: Runtime information about your Space including `stage=PAUSED` and requested hardware.

        Raises:
            [`~utils.RepositoryNotFoundError`]:
                If your Space is not found (error 404). Most probably wrong repo_id or your space is private but you
                are not authenticated.
            [`~utils.HfHubHTTPError`]:
                403 Forbidden: only the owner of a Space can pause it. If you want to manage a Space that you don't
                own, either ask the owner by opening a Discussion or duplicate the Space.
            [`~utils.BadRequestError`]:
                If your Space is a static Space. Static Spaces are always running and never billed. If you want to hide
                a static Space, you can set it to private.
        """
    r = get_session().post(f'{self.endpoint}/api/spaces/{repo_id}/pause', headers=self._build_hf_headers(token=token))
    hf_raise_for_status(r)
    return SpaceRuntime(r.json())
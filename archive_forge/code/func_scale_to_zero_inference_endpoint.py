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
def scale_to_zero_inference_endpoint(self, name: str, *, namespace: Optional[str]=None, token: Optional[str]=None) -> InferenceEndpoint:
    """Scale Inference Endpoint to zero.

        An Inference Endpoint scaled to zero will not be charged. It will be resume on the next request to it, with a
        cold start delay. This is different than pausing the Inference Endpoint with [`pause_inference_endpoint`], which
        would require a manual resume with [`resume_inference_endpoint`].

        For convenience, you can also scale an Inference Endpoint to zero using [`InferenceEndpoint.scale_to_zero`].

        Args:
            name (`str`):
                The name of the Inference Endpoint to scale to zero.
            namespace (`str`, *optional*):
                The namespace in which the Inference Endpoint is located. Defaults to the current user.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token).

        Returns:
            [`InferenceEndpoint`]: information about the scaled-to-zero Inference Endpoint.
        """
    namespace = namespace or self._get_namespace(token=token)
    response = get_session().post(f'{INFERENCE_ENDPOINTS_ENDPOINT}/endpoint/{namespace}/{name}/scale-to-zero', headers=self._build_hf_headers(token=token))
    hf_raise_for_status(response)
    return InferenceEndpoint.from_raw(response.json(), namespace=namespace, token=token)
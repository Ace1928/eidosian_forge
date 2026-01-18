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
def parse_safetensors_file_metadata(self, repo_id: str, filename: str, *, repo_type: Optional[str]=None, revision: Optional[str]=None, token: Optional[str]=None) -> SafetensorsFileMetadata:
    """
        Parse metadata from a safetensors file on the Hub.

        To parse metadata from all safetensors files in a repo at once, use [`get_safetensors_metadata`].

        For more details regarding the safetensors format, check out https://huggingface.co/docs/safetensors/index#format.

        Args:
            repo_id (`str`):
                A user or an organization name and a repo name separated by a `/`.
            filename (`str`):
                The name of the file in the repo.
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if the file is in a dataset or space, `None` or `"model"` if in a
                model. Default is `None`.
            revision (`str`, *optional*):
                The git revision to fetch the file from. Can be a branch name, a tag, or a commit hash. Defaults to the
                head of the `"main"` branch.
            token (`bool` or `str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token). If `None` or `True` and
                machine is logged in (through `huggingface-cli login` or [`~huggingface_hub.login`]), token will be
                retrieved from the cache. If `False`, token is not sent in the request header.

        Returns:
            [`SafetensorsFileMetadata`]: information related to a safetensors file.

        Raises:
            - [`NotASafetensorsRepoError`]: if the repo is not a safetensors repo i.e. doesn't have either a
              `model.safetensors` or a `model.safetensors.index.json` file.
            - [`SafetensorsParsingError`]: if a safetensors file header couldn't be parsed correctly.
        """
    url = hf_hub_url(repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision, endpoint=self.endpoint)
    _headers = self._build_hf_headers(token=token)
    response = get_session().get(url, headers={**_headers, 'range': 'bytes=0-100000'})
    hf_raise_for_status(response)
    metadata_size = struct.unpack('<Q', response.content[:8])[0]
    if metadata_size > SAFETENSORS_MAX_HEADER_LENGTH:
        raise SafetensorsParsingError(f"Failed to parse safetensors header for '{filename}' (repo '{repo_id}', revision '{revision or DEFAULT_REVISION}'): safetensors header is too big. Maximum supported size is {SAFETENSORS_MAX_HEADER_LENGTH} bytes (got {metadata_size}).")
    if metadata_size <= 100000:
        metadata_as_bytes = response.content[8:8 + metadata_size]
    else:
        response = get_session().get(url, headers={**_headers, 'range': f'bytes=8-{metadata_size + 7}'})
        hf_raise_for_status(response)
        metadata_as_bytes = response.content
    try:
        metadata_as_dict = json.loads(metadata_as_bytes.decode(errors='ignore'))
    except json.JSONDecodeError as e:
        raise SafetensorsParsingError(f"Failed to parse safetensors header for '{filename}' (repo '{repo_id}', revision '{revision or DEFAULT_REVISION}'): header is not json-encoded string. Please make sure this is a correctly formatted safetensors file.") from e
    try:
        return SafetensorsFileMetadata(metadata=metadata_as_dict.get('__metadata__', {}), tensors={key: TensorInfo(dtype=tensor['dtype'], shape=tensor['shape'], data_offsets=tuple(tensor['data_offsets'])) for key, tensor in metadata_as_dict.items() if key != '__metadata__'})
    except (KeyError, IndexError) as e:
        raise SafetensorsParsingError(f"Failed to parse safetensors header for '{filename}' (repo '{repo_id}', revision '{revision or DEFAULT_REVISION}'): header format not recognized. Please make sure this is a correctly formatted safetensors file.") from e
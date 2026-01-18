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
def create_pull_request(self, repo_id: str, title: str, *, token: Optional[str]=None, description: Optional[str]=None, repo_type: Optional[str]=None) -> DiscussionWithDetails:
    """Creates a Pull Request . Pull Requests created programmatically will be in `"draft"` status.

        Creating a Pull Request with changes can also be done at once with [`HfApi.create_commit`];

        This is a wrapper around [`HfApi.create_discussion`].

        Args:
            repo_id (`str`):
                A namespace (user or an organization) and a repo name separated
                by a `/`.
            title (`str`):
                The title of the discussion. It can be up to 200 characters long,
                and must be at least 3 characters long. Leading and trailing whitespaces
                will be stripped.
            token (`str`, *optional*):
                An authentication token (See https://huggingface.co/settings/token)
            description (`str`, *optional*):
                An optional description for the Pull Request.
                Defaults to `"Discussion opened with the huggingface_hub Python library"`
            repo_type (`str`, *optional*):
                Set to `"dataset"` or `"space"` if uploading to a dataset or
                space, `None` or `"model"` if uploading to a model. Default is
                `None`.

        Returns: [`DiscussionWithDetails`]

        <Tip>

        Raises the following errors:

            - [`HTTPError`](https://requests.readthedocs.io/en/latest/api/#requests.HTTPError)
              if the HuggingFace API returned an error
            - [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
              if some parameter value is invalid
            - [`~utils.RepositoryNotFoundError`]
              If the repository to download from cannot be found. This may be because it doesn't exist,
              or because it is set to `private` and you do not have access.

        </Tip>"""
    return self.create_discussion(repo_id=repo_id, title=title, token=token, description=description, repo_type=repo_type, pull_request=True)
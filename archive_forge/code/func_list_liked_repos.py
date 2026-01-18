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
def list_liked_repos(self, user: Optional[str]=None, *, token: Optional[str]=None) -> UserLikes:
    """
        List all public repos liked by a user on huggingface.co.

        This list is public so token is optional. If `user` is not passed, it defaults to
        the logged in user.

        See also [`like`] and [`unlike`].

        Args:
            user (`str`, *optional*):
                Name of the user for which you want to fetch the likes.
            token (`str`, *optional*):
                A valid authentication token (see https://huggingface.co/settings/token).
                Used only if `user` is not passed to implicitly determine the current
                user name.

        Returns:
            [`UserLikes`]: object containing the user name and 3 lists of repo ids (1 for
            models, 1 for datasets and 1 for Spaces).

        Raises:
            [`ValueError`](https://docs.python.org/3/library/exceptions.html#ValueError)
                If `user` is not passed and no token found (either from argument or from machine).

        Example:
        ```python
        >>> from huggingface_hub import list_liked_repos

        >>> likes = list_liked_repos("julien-c")

        >>> likes.user
        "julien-c"

        >>> likes.models
        ["osanseviero/streamlit_1.15", "Xhaheen/ChatGPT_HF", ...]
        ```
        """
    if user is None:
        me = self.whoami(token=token)
        if me['type'] == 'user':
            user = me['name']
        else:
            raise ValueError("Cannot list liked repos. You must provide a 'user' as input or be logged in as a user.")
    path = f'{self.endpoint}/api/users/{user}/likes'
    headers = self._build_hf_headers(token=token)
    likes = list(paginate(path, params={}, headers=headers))
    return UserLikes(user=user, total=len(likes), models=[like['repo']['name'] for like in likes if like['repo']['type'] == 'model'], datasets=[like['repo']['name'] for like in likes if like['repo']['type'] == 'dataset'], spaces=[like['repo']['name'] for like in likes if like['repo']['type'] == 'space'])
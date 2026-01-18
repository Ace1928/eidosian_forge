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
def delete_collection_item(self, collection_slug: str, item_object_id: str, *, missing_ok: bool=False, token: Optional[str]=None) -> None:
    """Delete an item from a collection.

        Args:
            collection_slug (`str`):
                Slug of the collection to update. Example: `"TheBloke/recent-models-64f9a55bb3115b4f513ec026"`.
            item_object_id (`str`):
                ID of the item in the collection. This is not the id of the item on the Hub (repo_id or paper id).
                It must be retrieved from a [`CollectionItem`] object. Example: `collection.items[0]._id`.
            missing_ok (`bool`, *optional*):
                If `True`, do not raise an error if item doesn't exists.
            token (`str`, *optional*):
                Hugging Face token. Will default to the locally saved token if not provided.

        Example:

        ```py
        >>> from huggingface_hub import get_collection, delete_collection_item

        # Get collection first
        >>> collection = get_collection("TheBloke/recent-models-64f9a55bb3115b4f513ec026")

        # Delete item based on its ID
        >>> delete_collection_item(
        ...     collection_slug="TheBloke/recent-models-64f9a55bb3115b4f513ec026",
        ...     item_object_id=collection.items[-1].item_object_id,
        ... )
        ```
        """
    r = get_session().delete(f'{self.endpoint}/api/collections/{collection_slug}/items/{item_object_id}', headers=self._build_hf_headers(token=token))
    try:
        hf_raise_for_status(r)
    except HTTPError as err:
        if missing_ok and err.response.status_code == 404:
            return
        else:
            raise
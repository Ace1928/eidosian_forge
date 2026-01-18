import hashlib
import math
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union
from urllib.parse import quote
import requests
import urllib3
from wandb.errors.term import termwarn
from wandb.sdk.artifacts.artifact_file_cache import (
from wandb.sdk.artifacts.storage_handlers.azure_handler import AzureHandler
from wandb.sdk.artifacts.storage_handlers.gcs_handler import GCSHandler
from wandb.sdk.artifacts.storage_handlers.http_handler import HTTPHandler
from wandb.sdk.artifacts.storage_handlers.local_file_handler import LocalFileHandler
from wandb.sdk.artifacts.storage_handlers.multi_handler import MultiHandler
from wandb.sdk.artifacts.storage_handlers.s3_handler import S3Handler
from wandb.sdk.artifacts.storage_handlers.tracking_handler import TrackingHandler
from wandb.sdk.artifacts.storage_handlers.wb_artifact_handler import WBArtifactHandler
from wandb.sdk.artifacts.storage_handlers.wb_local_artifact_handler import (
from wandb.sdk.artifacts.storage_layout import StorageLayout
from wandb.sdk.artifacts.storage_policies.register import WANDB_STORAGE_POLICY
from wandb.sdk.artifacts.storage_policy import StoragePolicy
from wandb.sdk.internal.internal_api import Api as InternalApi
from wandb.sdk.internal.thread_local_settings import _thread_local_api_settings
from wandb.sdk.lib.hashutil import B64MD5, b64_to_hex_id, hex_to_b64_id
from wandb.sdk.lib.paths import FilePathStr, URIStr
def load_reference(self, manifest_entry: 'ArtifactManifestEntry', local: bool=False, dest_path: Optional[str]=None) -> Union[FilePathStr, URIStr]:
    assert manifest_entry.ref is not None
    used_handler = self._handler._get_handler(manifest_entry.ref)
    if hasattr(used_handler, '_cache'):
        used_handler._cache._override_cache_path = dest_path
    return self._handler.load_path(manifest_entry, local)
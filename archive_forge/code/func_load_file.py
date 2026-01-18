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
def load_file(self, artifact: 'Artifact', manifest_entry: 'ArtifactManifestEntry', dest_path: Optional[str]=None) -> FilePathStr:
    self._cache._override_cache_path = dest_path
    path, hit, cache_open = self._cache.check_md5_obj_path(B64MD5(manifest_entry.digest), manifest_entry.size if manifest_entry.size is not None else 0)
    if hit:
        return path
    if manifest_entry._download_url is not None:
        response = self._session.get(manifest_entry._download_url, stream=True)
        try:
            response.raise_for_status()
        except Exception:
            manifest_entry._download_url = None
    if manifest_entry._download_url is None:
        auth = None
        if not _thread_local_api_settings.cookies:
            assert self._api.api_key is not None
            auth = ('api', self._api.api_key)
        response = self._session.get(self._file_url(self._api, artifact.entity, manifest_entry), auth=auth, cookies=_thread_local_api_settings.cookies, headers=_thread_local_api_settings.headers, stream=True)
        response.raise_for_status()
    with cache_open(mode='wb') as file:
        for data in response.iter_content(chunk_size=16 * 1024):
            file.write(data)
    return path
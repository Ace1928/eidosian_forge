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
def s3_multipart_file_upload(self, file_path: str, chunk_size: int, hex_digests: Dict[int, str], multipart_urls: Dict[int, str], extra_headers: Dict[str, str]) -> List[Dict[str, Any]]:
    etags = []
    part_number = 1
    with open(file_path, 'rb') as f:
        while True:
            data = f.read(chunk_size)
            if not data:
                break
            md5_b64_str = str(hex_to_b64_id(hex_digests[part_number]))
            upload_resp = self._api.upload_multipart_file_chunk_retry(multipart_urls[part_number], data, extra_headers={'content-md5': md5_b64_str, 'content-length': str(len(data)), 'content-type': extra_headers.get('Content-Type', '')})
            assert upload_resp is not None
            etags.append({'partNumber': part_number, 'hexMD5': upload_resp.headers['ETag']})
            part_number += 1
    return etags
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
def store_file_sync(self, artifact_id: str, artifact_manifest_id: str, entry: 'ArtifactManifestEntry', preparer: 'StepPrepare', progress_callback: Optional['progress.ProgressFn']=None) -> bool:
    """Upload a file to the artifact store.

        Returns:
            True if the file was a duplicate (did not need to be uploaded),
            False if it needed to be uploaded or was a reference (nothing to dedupe).
        """
    file_size = entry.size if entry.size is not None else 0
    chunk_size = self.calc_chunk_size(file_size)
    upload_parts = []
    hex_digests = {}
    file_path = entry.local_path if entry.local_path is not None else ''
    if file_size >= S3_MIN_MULTI_UPLOAD_SIZE and file_size <= S3_MAX_MULTI_UPLOAD_SIZE:
        part_number = 1
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(chunk_size)
                if not data:
                    break
                hex_digest = hashlib.md5(data).hexdigest()
                upload_parts.append({'hexMD5': hex_digest, 'partNumber': part_number})
                hex_digests[part_number] = hex_digest
                part_number += 1
    resp = preparer.prepare_sync({'artifactID': artifact_id, 'artifactManifestID': artifact_manifest_id, 'name': entry.path, 'md5': entry.digest, 'uploadPartsInput': upload_parts}).get()
    entry.birth_artifact_id = resp.birth_artifact_id
    multipart_urls = resp.multipart_upload_urls
    if resp.upload_url is None:
        return True
    if entry.local_path is None:
        return False
    extra_headers = {header.split(':', 1)[0]: header.split(':', 1)[1] for header in resp.upload_headers or {}}
    if multipart_urls is None and resp.upload_url:
        self.default_file_upload(resp.upload_url, file_path, extra_headers, progress_callback)
    else:
        if multipart_urls is None:
            raise ValueError(f'No multipart urls to upload for file: {file_path}')
        etags = self.s3_multipart_file_upload(file_path, chunk_size, hex_digests, multipart_urls, extra_headers)
        assert resp.storage_path is not None
        self._api.complete_multipart_upload_artifact(artifact_id, resp.storage_path, etags, resp.upload_id)
    self._write_cache(entry)
    return False
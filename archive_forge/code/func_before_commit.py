import concurrent.futures
import json
import os
import sys
import tempfile
from typing import TYPE_CHECKING, Awaitable, Dict, Optional, Sequence
import wandb
import wandb.filesync.step_prepare
from wandb import util
from wandb.sdk.artifacts.artifact_manifest import ArtifactManifest
from wandb.sdk.artifacts.staging import get_staging_dir
from wandb.sdk.lib.hashutil import B64MD5, b64_to_hex_id, md5_file_b64
from wandb.sdk.lib.paths import URIStr
def before_commit() -> None:
    self._resolve_client_id_manifest_references()
    with tempfile.NamedTemporaryFile('w+', suffix='.json', delete=False) as fp:
        path = os.path.abspath(fp.name)
        json.dump(self._manifest.to_manifest_json(), fp, indent=4)
    digest = md5_file_b64(path)
    if distributed_id or incremental:
        _, resp = self._api.update_artifact_manifest(artifact_manifest_id, digest=digest)
    else:
        _, resp = self._api.create_artifact_manifest(manifest_filename, digest, artifact_id, base_artifact_id=base_id)
    upload_url = resp['uploadUrl']
    upload_headers = resp['uploadHeaders']
    extra_headers = {}
    for upload_header in upload_headers:
        key, val = upload_header.split(':', 1)
        extra_headers[key] = val
    with open(path, 'rb') as fp2:
        self._api.upload_file_retry(upload_url, fp2, extra_headers=extra_headers)
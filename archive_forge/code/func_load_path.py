import time
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Dict, Optional, Sequence, Tuple, Union
from urllib.parse import ParseResult, urlparse
from wandb import util
from wandb.errors.term import termlog
from wandb.sdk.artifacts.artifact_file_cache import get_artifact_file_cache
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
from wandb.sdk.artifacts.storage_handler import DEFAULT_MAX_OBJECTS, StorageHandler
from wandb.sdk.lib.hashutil import B64MD5
from wandb.sdk.lib.paths import FilePathStr, StrPath, URIStr
def load_path(self, manifest_entry: ArtifactManifestEntry, local: bool=False) -> Union[URIStr, FilePathStr]:
    if not local:
        assert manifest_entry.ref is not None
        return manifest_entry.ref
    path, hit, cache_open = self._cache.check_md5_obj_path(B64MD5(manifest_entry.digest), manifest_entry.size if manifest_entry.size is not None else 0)
    if hit:
        return path
    self.init_gcs()
    assert self._client is not None
    assert manifest_entry.ref is not None
    bucket, key, _ = self._parse_uri(manifest_entry.ref)
    version = manifest_entry.extra.get('versionID')
    obj = None
    if version is not None:
        obj = self._client.bucket(bucket).get_blob(key, generation=version)
    if obj is None:
        obj = self._client.bucket(bucket).get_blob(key)
        if obj is None:
            raise ValueError(f'Unable to download object {manifest_entry.ref} with generation {version}')
        md5 = obj.md5_hash
        if md5 != manifest_entry.digest:
            raise ValueError(f'Digest mismatch for object {manifest_entry.ref}: expected {manifest_entry.digest} but found {md5}')
    with cache_open(mode='wb') as f:
        obj.download_to_file(f)
    return path
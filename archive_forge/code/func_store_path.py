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
def store_path(self, artifact: 'Artifact', path: Union[URIStr, FilePathStr], name: Optional[StrPath]=None, checksum: bool=True, max_objects: Optional[int]=None) -> Sequence[ArtifactManifestEntry]:
    self.init_gcs()
    assert self._client is not None
    bucket, key, version = self._parse_uri(path)
    path = URIStr(f'{self._scheme}://{bucket}/{key}')
    max_objects = max_objects or DEFAULT_MAX_OBJECTS
    if not checksum:
        return [ArtifactManifestEntry(path=name or key, ref=path, digest=path)]
    start_time = None
    obj = self._client.bucket(bucket).get_blob(key, generation=version)
    if obj is None and version is not None:
        raise ValueError(f'Object does not exist: {path}#{version}')
    multi = obj is None
    if multi:
        start_time = time.time()
        termlog('Generating checksum for up to %i objects with prefix "%s"... ' % (max_objects, key), newline=False)
        objects = self._client.bucket(bucket).list_blobs(prefix=key, max_results=max_objects)
    else:
        objects = [obj]
    entries = [self._entry_from_obj(obj, path, name, prefix=key, multi=multi) for obj in objects]
    if start_time is not None:
        termlog('Done. %.1fs' % (time.time() - start_time), prefix=False)
    if len(entries) > max_objects:
        raise ValueError('Exceeded %i objects tracked, pass max_objects to add_reference' % max_objects)
    return entries
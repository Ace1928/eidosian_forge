from typing import Any, Dict, Mapping, Optional
from wandb.sdk.artifacts.artifact_manifest import ArtifactManifest
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry
from wandb.sdk.artifacts.storage_policy import StoragePolicy
from wandb.sdk.internal.internal_api import Api as InternalApi
from wandb.sdk.lib.hashutil import HexMD5, _md5
@classmethod
def from_manifest_json(cls, manifest_json: Dict, api: Optional[InternalApi]=None) -> 'ArtifactManifestV1':
    if manifest_json['version'] != cls.version():
        raise ValueError('Expected manifest version 1, got %s' % manifest_json['version'])
    storage_policy_name = manifest_json['storagePolicy']
    storage_policy_config = manifest_json.get('storagePolicyConfig', {})
    storage_policy_cls = StoragePolicy.lookup_by_name(storage_policy_name)
    entries: Mapping[str, ArtifactManifestEntry]
    entries = {name: ArtifactManifestEntry(path=name, digest=val['digest'], birth_artifact_id=val.get('birthArtifactID'), ref=val.get('ref'), size=val.get('size'), extra=val.get('extra'), local_path=val.get('local_path')) for name, val in manifest_json['contents'].items()}
    return cls(storage_policy_cls.from_config(storage_policy_config, api=api), entries)
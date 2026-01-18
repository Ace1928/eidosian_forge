from typing import TYPE_CHECKING, Dict, Optional, Sequence, Type, Union
from wandb.sdk.internal.internal_api import Api as InternalApi
from wandb.sdk.lib.paths import FilePathStr, URIStr
@classmethod
def lookup_by_name(cls, name: str) -> Type['StoragePolicy']:
    import wandb.sdk.artifacts.storage_policies
    for sub in cls.__subclasses__():
        if sub.name() == name:
            return sub
    raise NotImplementedError(f"Failed to find storage policy '{name}'")
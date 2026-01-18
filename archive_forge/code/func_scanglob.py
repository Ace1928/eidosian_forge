from __future__ import annotations
import concurrent.futures
from typing import (
import urllib3
from blobfile._common import DirEntry, Stat, RemoteOrLocalPath
from blobfile._context import (
def scanglob(pattern: str, parallel: bool=False, shard_prefix_length: int=0) -> Iterator[DirEntry]:
    """
    Same as `glob`, but returns `DirEntry` objects instead of strings
    """
    return default_context.scanglob(pattern=pattern, parallel=parallel, shard_prefix_length=shard_prefix_length)
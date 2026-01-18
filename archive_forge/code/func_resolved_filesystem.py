import posixpath
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Callable, Dict, List, Optional
from ray.util.annotations import DeveloperAPI, PublicAPI
@property
def resolved_filesystem(self) -> 'pyarrow.fs.FileSystem':
    """Returns the filesystem resolved for compatibility with a base directory."""
    if self._resolved_filesystem is None:
        self._normalize_base_dir()
    return self._resolved_filesystem
from __future__ import absolute_import
import pathlib
import contextlib
from typing import Optional, Union, Dict, TYPE_CHECKING
from lazyops.types.models import BaseSettings
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
from lazyops.imports._aiokeydb import (
@lazyproperty
def keydb(self) -> 'aiokeydb.KeyDBSettings':
    if not _aiokeydb_available:
        resolve_aiokeydb()
    return aiokeydb.KeyDBClient.get_settings()
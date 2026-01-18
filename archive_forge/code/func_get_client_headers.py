from __future__ import absolute_import
import pathlib
import contextlib
from typing import Optional, Union, Dict, TYPE_CHECKING
from lazyops.types.models import BaseSettings
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
from lazyops.imports._aiokeydb import (
def get_client_headers(self) -> Dict[str, str]:
    return self.client.get_headers()
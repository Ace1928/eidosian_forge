from __future__ import absolute_import
import pathlib
import contextlib
from typing import Optional, Union, Dict, TYPE_CHECKING
from lazyops.types.models import BaseSettings
from lazyops.types.classprops import lazyproperty
from lazyops.imports._fileio import (
from lazyops.imports._aiokeydb import (
@lazyproperty
def client_endpoint(self) -> str:
    """
        Returns the client endpoint
        """
    return self.client.endpoints[self.default_client_name] or self.client.endpoints['local']
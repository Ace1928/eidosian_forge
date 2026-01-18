from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import (
@property
def max_staleness(self) -> int:
    """The maximum estimated length of time (in seconds) a replica set
        secondary can fall behind the primary in replication before it will
        no longer be selected for operations, or -1 for no maximum.
        """
    return self.__max_staleness
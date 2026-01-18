from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import (
def read_pref_mode_from_name(name: str) -> int:
    """Get the read preference mode from mongos/uri name."""
    return _MONGOS_MODES.index(name)
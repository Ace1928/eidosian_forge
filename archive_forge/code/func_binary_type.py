from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Type
import bson
from bson.binary import Binary
from bson.son import SON
from pymongo.errors import ConfigurationError, OperationFailure
def binary_type(self) -> Type[Binary]:
    """Return the bson.binary.Binary type."""
    return Binary
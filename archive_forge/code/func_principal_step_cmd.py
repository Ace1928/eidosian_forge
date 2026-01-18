from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, MutableMapping, Optional
import bson
from bson.binary import Binary
from bson.son import SON
from pymongo.errors import ConfigurationError, OperationFailure
def principal_step_cmd(self) -> SON[str, Any]:
    """Get a SASL start command with an optional principal name"""
    payload = {}
    principal_name = self.username
    if principal_name:
        payload['n'] = principal_name
    return SON([('saslStart', 1), ('mechanism', 'MONGODB-OIDC'), ('payload', Binary(bson.encode(payload))), ('autoAuthorize', 1)])
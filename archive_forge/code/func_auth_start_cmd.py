from __future__ import annotations
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Mapping, MutableMapping, Optional
import bson
from bson.binary import Binary
from bson.son import SON
from pymongo.errors import ConfigurationError, OperationFailure
def auth_start_cmd(self, use_callback: bool=True) -> Optional[SON[str, Any]]:
    if self.idp_info is None:
        return self.principal_step_cmd()
    token = self.get_current_token(use_callback)
    if not token:
        return None
    bin_payload = Binary(bson.encode({'jwt': token}))
    return SON([('saslStart', 1), ('mechanism', 'MONGODB-OIDC'), ('payload', bin_payload)])
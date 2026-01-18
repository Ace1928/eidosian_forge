from __future__ import annotations
import copy
from collections.abc import MutableMapping
from itertools import islice
from typing import (
from bson.objectid import ObjectId
from bson.raw_bson import RawBSONDocument
from bson.son import SON
from pymongo import _csot, common
from pymongo.client_session import ClientSession, _validate_session_write_concern
from pymongo.common import (
from pymongo.errors import (
from pymongo.helpers import _RETRYABLE_ERROR_CODES, _get_wce_doc
from pymongo.message import (
from pymongo.read_preferences import ReadPreference
from pymongo.write_concern import WriteConcern
def add_replace(self, selector: Mapping[str, Any], replacement: Mapping[str, Any], upsert: bool=False, collation: Optional[Mapping[str, Any]]=None, hint: Union[str, SON[str, Any], None]=None) -> None:
    """Create a replace document and add it to the list of ops."""
    validate_ok_for_replace(replacement)
    cmd = SON([('q', selector), ('u', replacement), ('multi', False), ('upsert', upsert)])
    if collation is not None:
        self.uses_collation = True
        cmd['collation'] = collation
    if hint is not None:
        self.uses_hint_update = True
        cmd['hint'] = hint
    self.ops.append((_UPDATE, cmd))
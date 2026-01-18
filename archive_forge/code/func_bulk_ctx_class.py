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
@property
def bulk_ctx_class(self) -> Type[_BulkWriteContext]:
    encrypter = self.collection.database.client._encrypter
    if encrypter and (not encrypter._bypass_auto_encryption):
        return _EncryptedBulkWriteContext
    else:
        return _BulkWriteContext
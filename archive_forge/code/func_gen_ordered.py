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
def gen_ordered(self) -> Iterator[Optional[_Run]]:
    """Generate batches of operations, batched by type of
        operation, in the order **provided**.
        """
    run = None
    for idx, (op_type, operation) in enumerate(self.ops):
        if run is None:
            run = _Run(op_type)
        elif run.op_type != op_type:
            yield run
            run = _Run(op_type)
        run.add(idx, operation)
    yield run
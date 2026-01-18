from __future__ import annotations
import copy
import warnings
from collections import deque
from typing import (
from bson import RE_TYPE, _convert_raw_document_lists_to_streams
from bson.code import Code
from bson.son import SON
from pymongo import helpers
from pymongo.collation import validate_collation_or_none
from pymongo.common import (
from pymongo.errors import ConnectionFailure, InvalidOperation, OperationFailure
from pymongo.lock import _create_lock
from pymongo.message import (
from pymongo.response import PinnedResponse
from pymongo.typings import _Address, _CollationIn, _DocumentOut, _DocumentType
@property
def retrieved(self) -> int:
    """The number of documents retrieved so far."""
    return self.__retrieved
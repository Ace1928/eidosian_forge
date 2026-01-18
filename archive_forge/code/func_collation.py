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
def collation(self, collation: Optional[_CollationIn]) -> Cursor[_DocumentType]:
    """Adds a :class:`~pymongo.collation.Collation` to this query.

        Raises :exc:`TypeError` if `collation` is not an instance of
        :class:`~pymongo.collation.Collation` or a ``dict``. Raises
        :exc:`~pymongo.errors.InvalidOperation` if this :class:`Cursor` has
        already been used. Only the last collation applied to this cursor has
        any effect.

        :Parameters:
          - `collation`: An instance of :class:`~pymongo.collation.Collation`.
        """
    self.__check_okay_to_chain()
    self.__collation = validate_collation_or_none(collation)
    return self
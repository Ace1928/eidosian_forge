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
def max_scan(self, max_scan: Optional[int]) -> Cursor[_DocumentType]:
    """**DEPRECATED** - Limit the number of documents to scan when
        performing the query.

        Raises :class:`~pymongo.errors.InvalidOperation` if this
        cursor has already been used. Only the last :meth:`max_scan`
        applied to this cursor has any effect.

        :Parameters:
          - `max_scan`: the maximum number of documents to scan

        .. versionchanged:: 3.7
          Deprecated :meth:`max_scan`. Support for this option is deprecated in
          MongoDB 4.0. Use :meth:`max_time_ms` instead to limit server side
          execution time.
        """
    self.__check_okay_to_chain()
    self.__max_scan = max_scan
    return self
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
def allow_disk_use(self, allow_disk_use: bool) -> Cursor[_DocumentType]:
    """Specifies whether MongoDB can use temporary disk files while
        processing a blocking sort operation.

        Raises :exc:`TypeError` if `allow_disk_use` is not a boolean.

        .. note:: `allow_disk_use` requires server version **>= 4.4**

        :Parameters:
          - `allow_disk_use`: if True, MongoDB may use temporary
            disk files to store data exceeding the system memory limit while
            processing a blocking sort operation.

        .. versionadded:: 3.11
        """
    if not isinstance(allow_disk_use, bool):
        raise TypeError('allow_disk_use must be a bool')
    self.__check_okay_to_chain()
    self.__allow_disk_use = allow_disk_use
    return self
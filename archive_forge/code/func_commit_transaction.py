from the same snapshot timestamp. The server chooses the latest
from __future__ import annotations
import collections
import time
import uuid
from collections.abc import Mapping as _Mapping
from typing import (
from bson.binary import Binary
from bson.int64 import Int64
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import _csot
from pymongo.cursor import _ConnectionManager
from pymongo.errors import (
from pymongo.helpers import _RETRYABLE_ERROR_CODES
from pymongo.read_concern import ReadConcern
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.server_type import SERVER_TYPE
from pymongo.write_concern import WriteConcern
def commit_transaction(self) -> None:
    """Commit a multi-statement transaction.

        .. versionadded:: 3.7
        """
    self._check_ended()
    state = self._transaction.state
    if state is _TxnState.NONE:
        raise InvalidOperation('No transaction started')
    elif state in (_TxnState.STARTING, _TxnState.COMMITTED_EMPTY):
        self._transaction.state = _TxnState.COMMITTED_EMPTY
        return
    elif state is _TxnState.ABORTED:
        raise InvalidOperation('Cannot call commitTransaction after calling abortTransaction')
    elif state is _TxnState.COMMITTED:
        self._transaction.state = _TxnState.IN_PROGRESS
    try:
        self._finish_transaction_with_retry('commitTransaction')
    except ConnectionFailure as exc:
        exc._remove_error_label('TransientTransactionError')
        _reraise_with_unknown_commit(exc)
    except WTimeoutError as exc:
        _reraise_with_unknown_commit(exc)
    except OperationFailure as exc:
        if exc.code not in _UNKNOWN_COMMIT_ERROR_CODES:
            raise
        _reraise_with_unknown_commit(exc)
    finally:
        self._transaction.state = _TxnState.COMMITTED
from __future__ import annotations
import atexit
import time
import weakref
from typing import TYPE_CHECKING, Any, Mapping, Optional, cast
from pymongo import common, periodic_executor
from pymongo._csot import MovingMinimum
from pymongo.errors import NotPrimaryError, OperationFailure, _OperationCancelled
from pymongo.hello import Hello
from pymongo.lock import _create_lock
from pymongo.periodic_executor import _shutdown_executors
from pymongo.pool import _is_faas
from pymongo.read_preferences import MovingAverage
from pymongo.server_description import ServerDescription
from pymongo.srv_resolver import _SrvResolver
def cancel_check(self) -> None:
    """Cancel any concurrent hello check.

        Note: this is called from a weakref.proxy callback and MUST NOT take
        any locks.
        """
    context = self._cancel_context
    if context:
        context.cancel()
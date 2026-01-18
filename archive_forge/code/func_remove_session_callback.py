from __future__ import annotations
import logging # isort:skip
import weakref
from collections import defaultdict
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable
from ..core.enums import HoldPolicy, HoldPolicyType
from ..events import (
from ..model import Model
from ..models.callbacks import Callback as JSEventCallback
from ..util.callback_manager import _check_callback
from .events import (
from .locking import UnlockedDocumentProxy
def remove_session_callback(self, callback_obj: SessionCallback) -> None:
    """ Remove a callback added earlier with ``add_periodic_callback``,
        ``add_timeout_callback``, or ``add_next_tick_callback``.

        Returns:
            None

        Raises:
            KeyError, if the callback was never added

        """
    try:
        callback_objs = [callback_obj]
        self._session_callbacks.remove(callback_obj)
    except KeyError:
        raise ValueError('callback already ran or was already removed, cannot be removed again')
    doc = self._document()
    if doc is None:
        return
    for callback_obj in callback_objs:
        self.trigger_on_change(SessionCallbackRemoved(doc, callback_obj))
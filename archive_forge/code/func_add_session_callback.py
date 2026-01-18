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
def add_session_callback(self, callback_obj: SessionCallback, callback: Callback, one_shot: bool) -> SessionCallback:
    """ Internal implementation for adding session callbacks.

        Args:
            callback_obj (SessionCallback) :
                A session callback object that wraps a callable and is
                passed to ``trigger_on_change``.

            callback (callable) :
                A callable to execute when session events happen.

            one_shot (bool) :
                Whether the callback should immediately auto-remove itself
                after one execution.

        Returns:
            SessionCallback : passed in as ``callback_obj``.

        Raises:
            ValueError, if the callback has been previously added

        """
    doc = self._document()
    if doc is None:
        raise RuntimeError('Attempting to add session callback to already-destroyed Document')
    if one_shot:

        @wraps(callback)
        def remove_then_invoke() -> None:
            if callback_obj in self._session_callbacks:
                self.remove_session_callback(callback_obj)
            return callback()
        actual_callback = remove_then_invoke
    else:
        actual_callback = callback
    callback_obj._callback = _wrap_with_curdoc(doc, actual_callback)
    self._session_callbacks.add(callback_obj)
    self.trigger_on_change(SessionCallbackAdded(doc, callback_obj))
    return callback_obj
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
def notify_event(self, model: Model, event: ModelEvent, callback_invoker: Invoker) -> None:
    """

        """
    doc = self._document()
    if doc is None:
        return
    invoke_with_curdoc(doc, callback_invoker)
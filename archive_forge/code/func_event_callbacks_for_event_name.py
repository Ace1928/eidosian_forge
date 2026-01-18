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
def event_callbacks_for_event_name(self, event_name: str) -> tuple[EventCallback, ...]:
    """ Return a tuple containing all current event callbacks for the given
        event name.

        Args:
            event_name (str) : the event name to look up callbacks for

        """
    return tuple(self._event_callbacks.get(event_name, []))
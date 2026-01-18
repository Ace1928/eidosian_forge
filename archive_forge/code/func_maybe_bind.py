from __future__ import annotations
from copy import copy
from typing import TYPE_CHECKING, Any, Callable, TypeVar
from .connection import maybe_channel
from .exceptions import NotBoundError
from .utils.functional import ChannelPromise
def maybe_bind(self: _MaybeChannelBoundType, channel: Channel | Connection) -> _MaybeChannelBoundType:
    """Bind instance to channel if not already bound."""
    if not self.is_bound and channel:
        self._channel = maybe_channel(channel)
        self.when_bound()
        self._is_bound = True
    return self
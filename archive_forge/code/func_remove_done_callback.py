from . import events
import asyncio
import contextvars
import enum
import typing
def remove_done_callback(self, cb: typing.Callable) -> int:
    original_len = len(self._callbacks)
    self._callbacks = [_cb for _cb in self._callbacks if _cb != cb]
    return original_len - len(self._callbacks)
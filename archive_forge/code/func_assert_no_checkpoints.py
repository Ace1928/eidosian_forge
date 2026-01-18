from __future__ import annotations
from contextlib import AbstractContextManager, contextmanager
from typing import TYPE_CHECKING
from .. import _core
def assert_no_checkpoints() -> AbstractContextManager[None]:
    """Use as a context manager to check that the code inside the ``with``
    block does not execute any :ref:`checkpoints <checkpoints>`.

    Raises:
      AssertionError: if a checkpoint was executed.

    Example:
      Synchronous code never contains any checkpoints, but we can double-check
      that::

         send_channel, receive_channel = trio.open_memory_channel(10)
         with trio.testing.assert_no_checkpoints():
             send_channel.send_nowait(None)

    """
    __tracebackhide__ = True
    return _assert_yields_or_not(False)
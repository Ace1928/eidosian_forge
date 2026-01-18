from __future__ import annotations
import contextlib
import inspect
import signal
import threading
from typing import TYPE_CHECKING, AsyncIterator, Callable, Iterator
import outcome
import pytest
from trio.testing import RaisesGroup
from ... import _core
from ..._abc import Instrument
from ..._timeouts import sleep
from ..._util import signal_raise
from ...testing import wait_all_tasks_blocked
def test_ki_disabled_in_del() -> None:

    def nestedfunction() -> bool:
        return _core.currently_ki_protected()

    def __del__() -> None:
        assert _core.currently_ki_protected()
        assert nestedfunction()

    @_core.disable_ki_protection
    def outerfunction() -> None:
        assert not _core.currently_ki_protected()
        assert not nestedfunction()
        __del__()
    __del__()
    outerfunction()
    assert nestedfunction()
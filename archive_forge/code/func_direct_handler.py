from __future__ import annotations
import signal
from typing import TYPE_CHECKING, NoReturn
import pytest
import trio
from trio.testing import RaisesGroup
from .. import _core
from .._signals import _signal_handler, get_pending_signal_count, open_signal_receiver
from .._util import signal_raise
def direct_handler(signo: int, frame: FrameType | None) -> None:
    delivered_directly.add(signo)
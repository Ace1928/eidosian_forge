from __future__ import annotations
import random
import socket as stdlib_socket
from contextlib import suppress
from typing import TYPE_CHECKING, Awaitable, Callable, Tuple, TypeVar
import pytest
import trio
from ... import _core
from ...testing import assert_checkpoints, wait_all_tasks_blocked
def drain_socket(sock: stdlib_socket.socket) -> None:
    try:
        while True:
            sock.recv(65536)
    except BlockingIOError:
        pass
from __future__ import annotations
import math
import sys
from abc import ABCMeta, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Mapping
from os import PathLike
from signal import Signals
from socket import AddressFamily, SocketKind, socket
from typing import (
@classmethod
@abstractmethod
def run_sync_from_thread(cls, func: Callable[[Unpack[PosArgsT]], T_Retval], args: tuple[Unpack[PosArgsT]], token: object) -> T_Retval:
    pass
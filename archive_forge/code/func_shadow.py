from __future__ import annotations
import atexit
import os
from threading import Lock
from typing import Any, Callable, Generic, TypeVar, overload
from warnings import warn
from weakref import WeakSet
import zmq
from zmq._typing import TypeAlias
from zmq.backend import Context as ContextBase
from zmq.constants import ContextOption, Errno, SocketOption
from zmq.error import ZMQError
from zmq.utils.interop import cast_int_addr
from .attrsettr import AttributeSetter, OptValT
from .socket import Socket, SyncSocket
@classmethod
def shadow(cls: type[_ContextType], address: int | zmq.Context) -> _ContextType:
    """Shadow an existing libzmq context

        address is a zmq.Context or an integer (or FFI pointer)
        representing the address of the libzmq context.

        .. versionadded:: 14.1

        .. versionadded:: 25
            Support for shadowing `zmq.Context` objects,
            instead of just integer addresses.
        """
    return cls(shadow=address)
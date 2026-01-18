from __future__ import annotations
import errno
import pickle
import random
import sys
from typing import (
from warnings import warn
import zmq
from zmq._typing import Literal, TypeAlias
from zmq.backend import Socket as SocketBase
from zmq.error import ZMQBindError, ZMQError
from zmq.utils import jsonapi
from zmq.utils.interop import cast_int_addr
from ..constants import SocketOption, SocketType, _OptType
from .attrsettr import AttributeSetter
from .poll import Poller
def get_monitor_socket(self: _SocketType, events: int | None=None, addr: str | None=None) -> _SocketType:
    """Return a connected PAIR socket ready to receive the event notifications.

        .. versionadded:: libzmq-4.0
        .. versionadded:: 14.0

        Parameters
        ----------
        events : int
            default: `zmq.EVENT_ALL`
            The bitmask defining which events are wanted.
        addr : str
            The optional endpoint for the monitoring sockets.

        Returns
        -------
        socket : zmq.Socket
            The PAIR socket, connected and ready to receive messages.
        """
    if zmq.zmq_version_info() < (4,):
        raise NotImplementedError('get_monitor_socket requires libzmq >= 4, have %s' % zmq.zmq_version())
    if self._monitor_socket:
        if self._monitor_socket.closed:
            self._monitor_socket = None
        else:
            return self._monitor_socket
    if addr is None:
        addr = f'inproc://monitor.s-{self.FD}'
    if events is None:
        events = zmq.EVENT_ALL
    self.monitor(addr, events)
    self._monitor_socket = self.context.socket(zmq.PAIR)
    self._monitor_socket.connect(addr)
    return self._monitor_socket
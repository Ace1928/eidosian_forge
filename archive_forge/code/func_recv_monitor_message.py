from __future__ import annotations
import struct
from typing import Awaitable, overload
import zmq
import zmq.asyncio
from zmq._typing import TypedDict
from zmq.error import _check_version
def recv_monitor_message(socket: zmq.Socket, flags: int=0) -> _MonitorMessage | Awaitable[_MonitorMessage]:
    """Receive and decode the given raw message from the monitoring socket and return a dict.

    Requires libzmq ≥ 4.0

    The returned dict will have the following entries:
      event : int
        the event id as described in `libzmq.zmq_socket_monitor`
      value : int
        the event value associated with the event, see `libzmq.zmq_socket_monitor`
      endpoint : str
        the affected endpoint

    .. versionchanged:: 23.1
        Support for async sockets added.
        When called with a async socket,
        returns an awaitable for the monitor message.

    Parameters
    ----------
    socket : zmq.Socket
        The PAIR socket (created by other.get_monitor_socket()) on which to recv the message
    flags : int
        standard zmq recv flags

    Returns
    -------
    event : dict
        event description as dict with the keys `event`, `value`, and `endpoint`.
    """
    _check_version((4, 0), 'libzmq event API')
    msg = socket.recv_multipart(flags)
    if isinstance(msg, Awaitable):
        return _parse_monitor_msg_async(msg)
    return parse_monitor_message(msg)
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
def bind_to_random_port(self: _SocketType, addr: str, min_port: int=49152, max_port: int=65536, max_tries: int=100) -> int:
    """Bind this socket to a random port in a range.

        If the port range is unspecified, the system will choose the port.

        Parameters
        ----------
        addr : str
            The address string without the port to pass to ``Socket.bind()``.
        min_port : int, optional
            The minimum port in the range of ports to try (inclusive).
        max_port : int, optional
            The maximum port in the range of ports to try (exclusive).
        max_tries : int, optional
            The maximum number of bind attempts to make.

        Returns
        -------
        port : int
            The port the socket was bound to.

        Raises
        ------
        ZMQBindError
            if `max_tries` reached before successful bind
        """
    if zmq.zmq_version_info() >= (3, 2) and min_port == 49152 and (max_port == 65536):
        self.bind('%s:*' % addr)
        url = cast(bytes, self.last_endpoint).decode('ascii', 'replace')
        _, port_s = url.rsplit(':', 1)
        return int(port_s)
    for i in range(max_tries):
        try:
            port = random.randrange(min_port, max_port)
            self.bind(f'{addr}:{port}')
        except ZMQError as exception:
            en = exception.errno
            if en == zmq.EADDRINUSE:
                continue
            elif sys.platform == 'win32' and en == errno.EACCES:
                continue
            else:
                raise
        else:
            return port
    raise ZMQBindError('Could not bind socket to random port.')
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
def recv_serialized(self, deserialize, flags=0, copy=True):
    """Receive a message with a custom deserialization function.

        .. versionadded:: 17

        Parameters
        ----------
        deserialize : callable
            The deserialization function to use.
            deserialize will be called with one argument: the list of frames
            returned by recv_multipart() and can return any object.
        flags : int, optional
            Any valid flags for :func:`Socket.recv`.
        copy : bool, optional
            Whether to recv bytes or Frame objects.

        Returns
        -------
        obj : object
            The object returned by the deserialization function.

        Raises
        ------
        ZMQError
            for any of the reasons :func:`~Socket.recv` might fail
        """
    frames = self.recv_multipart(flags=flags, copy=copy)
    return self._deserialize(frames, deserialize)
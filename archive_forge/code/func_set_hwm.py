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
def set_hwm(self, value: int) -> None:
    """Set the High Water Mark.

        On libzmq ≥ 3, this sets both SNDHWM and RCVHWM


        .. warning::

            New values only take effect for subsequent socket
            bind/connects.
        """
    major = zmq.zmq_version_info()[0]
    if major >= 3:
        raised = None
        try:
            self.sndhwm = value
        except Exception as e:
            raised = e
        try:
            self.rcvhwm = value
        except Exception as e:
            raised = e
        if raised:
            raise raised
    else:
        self.set(zmq.HWM, value)
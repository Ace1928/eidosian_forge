from __future__ import annotations
import time
import warnings
from threading import Event
from weakref import ref
import cython as C
from cython import (
from cython.cimports.cpython import (
from cython.cimports.libc.errno import EAGAIN, EINTR, ENAMETOOLONG, ENOENT, ENOTSOCK
from cython.cimports.libc.stdint import uint32_t
from cython.cimports.libc.stdio import fprintf
from cython.cimports.libc.stdio import stderr as cstderr
from cython.cimports.libc.stdlib import free, malloc
from cython.cimports.libc.string import memcpy
from cython.cimports.zmq.backend.cython._externs import (
from cython.cimports.zmq.backend.cython.libzmq import (
from cython.cimports.zmq.backend.cython.libzmq import zmq_errno as _zmq_errno
from cython.cimports.zmq.backend.cython.libzmq import zmq_poll as zmq_poll_c
from cython.cimports.zmq.utils.buffers import asbuffer_r
import zmq
from zmq.constants import SocketOption, _OptType
from zmq.error import InterruptedSystemCall, ZMQError, _check_version
def fast_copy(self) -> Frame:
    new_msg: Frame = Frame()
    zmq_msg_copy(address(new_msg.zmq_msg), address(self.zmq_msg))
    if self._data is not None:
        new_msg._data = self._data
    if self._buffer is not None:
        new_msg._buffer = self._buffer
    if self._bytes is not None:
        new_msg._bytes = self._bytes
    new_msg.tracker_event = self.tracker_event
    new_msg.tracker = self.tracker
    return new_msg
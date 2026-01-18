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
def proxy_steerable(frontend: Socket, backend: Socket, capture: Socket=None, control: Socket=None):
    """
    Start a zeromq proxy with control flow.

    .. versionadded:: libzmq-4.1
    .. versionadded:: 18.0

    Parameters
    ----------
    frontend : Socket
        The Socket instance for the incoming traffic.
    backend : Socket
        The Socket instance for the outbound traffic.
    capture : Socket (optional)
        The Socket instance for capturing traffic.
    control : Socket (optional)
        The Socket instance for control flow.
    """
    rc: C.int = 0
    capture_handle: p_void
    if isinstance(capture, Socket):
        capture_handle = capture.handle
    else:
        capture_handle = NULL
    if isinstance(control, Socket):
        control_handle = control.handle
    else:
        control_handle = NULL
    while True:
        with nogil:
            rc = zmq_proxy_steerable(frontend.handle, backend.handle, capture_handle, control_handle)
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            continue
        else:
            break
    return rc
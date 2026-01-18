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
def monitored_queue(in_socket: Socket, out_socket: Socket, mon_socket: Socket, in_prefix: bytes=b'in', out_prefix: bytes=b'out'):
    """
    Start a monitored queue device.

    A monitored queue is very similar to the zmq.proxy device (monitored queue came first).

    Differences from zmq.proxy:

    - monitored_queue supports both in and out being ROUTER sockets
      (via swapping IDENTITY prefixes).
    - monitor messages are prefixed, making in and out messages distinguishable.

    Parameters
    ----------
    in_socket : zmq.Socket
        One of the sockets to the Queue. Its messages will be prefixed with
        'in'.
    out_socket : zmq.Socket
        One of the sockets to the Queue. Its messages will be prefixed with
        'out'. The only difference between in/out socket is this prefix.
    mon_socket : zmq.Socket
        This socket sends out every message received by each of the others
        with an in/out prefix specifying which one it was.
    in_prefix : str
        Prefix added to broadcast messages from in_socket.
    out_prefix : str
        Prefix added to broadcast messages from out_socket.
    """
    ins: p_void = in_socket.handle
    outs: p_void = out_socket.handle
    mons: p_void = mon_socket.handle
    in_msg = declare(zmq_msg_t)
    out_msg = declare(zmq_msg_t)
    swap_ids: bint
    msg_c: p_char = NULL
    msg_c_len = declare(Py_ssize_t)
    rc: C.int
    swap_ids = in_socket.type == ZMQ_ROUTER and out_socket.type == ZMQ_ROUTER
    asbuffer_r(in_prefix, cast(pointer(p_void), address(msg_c)), address(msg_c_len))
    rc = zmq_msg_init_size(address(in_msg), msg_c_len)
    _check_rc(rc)
    memcpy(zmq_msg_data(address(in_msg)), msg_c, zmq_msg_size(address(in_msg)))
    asbuffer_r(out_prefix, cast(pointer(p_void), address(msg_c)), address(msg_c_len))
    rc = zmq_msg_init_size(address(out_msg), msg_c_len)
    _check_rc(rc)
    while True:
        with nogil:
            memcpy(zmq_msg_data(address(out_msg)), msg_c, zmq_msg_size(address(out_msg)))
            rc = _mq_inline(ins, outs, mons, address(in_msg), address(out_msg), swap_ids)
        try:
            _check_rc(rc)
        except InterruptedSystemCall:
            continue
        else:
            break
    return rc
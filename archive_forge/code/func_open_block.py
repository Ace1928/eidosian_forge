import errno
import os
import socket
import sys
import ovs.poller
import ovs.socket_util
import ovs.vlog
@staticmethod
def open_block(error_stream, timeout=None):
    """Blocks until a Stream completes its connection attempt, either
        succeeding or failing, but no more than 'timeout' milliseconds.
        (error, stream) should be the tuple returned by Stream.open().
        Negative value of 'timeout' means infinite waiting.
        Returns a tuple of the same form.

        Typical usage:
        error, stream = Stream.open_block(Stream.open("unix:/tmp/socket"))"""
    error, stream = error_stream
    if not error:
        deadline = None
        if timeout is not None and timeout >= 0:
            deadline = ovs.timeval.msec() + timeout
        while True:
            error = stream.connect()
            if sys.platform == 'win32' and error == errno.WSAEWOULDBLOCK:
                error = errno.EAGAIN
            if error != errno.EAGAIN:
                break
            if deadline is not None and ovs.timeval.msec() > deadline:
                error = errno.ETIMEDOUT
                break
            stream.run()
            poller = ovs.poller.Poller()
            stream.run_wait(poller)
            stream.connect_wait(poller)
            if deadline is not None:
                poller.timer_wait_until(deadline)
            poller.block()
        if stream.socket is not None:
            assert error != errno.EINPROGRESS
    if error and stream:
        stream.close()
        stream = None
    return (error, stream)
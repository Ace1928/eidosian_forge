from __future__ import absolute_import
import errno
import logging
import select
import socket
import time
from serial.serialutil import SerialBase, SerialException, to_bytes, \
def reset_input_buffer(self):
    """Clear input buffer, discarding all that is in the buffer."""
    if not self.is_open:
        raise PortNotOpenError()
    ready = True
    while ready:
        ready, _, _ = select.select([self._socket], [], [], 0)
        try:
            if ready:
                ready = self._socket.recv(4096)
        except OSError as e:
            if e.errno not in (errno.EAGAIN, errno.EALREADY, errno.EWOULDBLOCK, errno.EINPROGRESS, errno.EINTR):
                raise SerialException('read failed: {}'.format(e))
        except (select.error, socket.error) as e:
            if e[0] not in (errno.EAGAIN, errno.EALREADY, errno.EWOULDBLOCK, errno.EINPROGRESS, errno.EINTR):
                raise SerialException('read failed: {}'.format(e))
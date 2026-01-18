import contextlib
import logging
import os
import socket
import sys
def onready(notify_socket, timeout):
    """Wait for systemd style notification on the socket.

    :param notify_socket: local socket address
    :type notify_socket:  string
    :param timeout:       socket timeout
    :type timeout:        float
    :returns:             0 service ready
                          1 service not ready
                          2 timeout occurred
    """
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    sock.bind(_abstractify(notify_socket))
    with contextlib.closing(sock):
        try:
            msg = sock.recv(512)
        except socket.timeout:
            return 2
        if b'READY=1' == msg:
            return 0
        else:
            return 1
from __future__ import print_function
import logging
import os
import random
import socket
import sys
import time
def get_port_from_port_server(portserver_address, pid=None):
    """Request a free a port from a system-wide portserver.

    This follows a very simple portserver protocol:
    The request consists of our pid (in ASCII) followed by a newline.
    The response is a port number and a newline, 0 on failure.

    This function is an implementation detail of pick_unused_port().
    It should not normally be called by code outside of this module.

    Args:
      portserver_address: The address (path) of a unix domain socket
        with which to connect to the portserver.  A leading '@'
        character indicates an address in the "abstract namespace."
        On systems without socket.AF_UNIX, this is an AF_INET address.
      pid: The PID to tell the portserver to associate the reservation with.
        If None, the current process's PID is used.

    Returns:
      The port number on success or None on failure.
    """
    if not portserver_address:
        return None
    if pid is None:
        pid = os.getpid()
    if _winapi:
        buf = _windows_get_port_from_port_server(portserver_address, pid)
    else:
        buf = _posix_get_port_from_port_server(portserver_address, pid)
    if buf is None:
        return None
    try:
        port = int(buf.split(b'\n')[0])
    except ValueError:
        print('Portserver failed to find a port.', file=sys.stderr)
        return None
    _owned_ports.add(port)
    return port
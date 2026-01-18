import atexit
import os
import re
import signal
import socket
import sys
import warnings
from getpass import getpass, getuser
from multiprocessing import Process
def open_tunnel(addr, server, keyfile=None, password=None, paramiko=None, timeout=60):
    """Open a tunneled connection from a 0MQ url.

    For use inside tunnel_connection.

    Returns
    -------

    (url, tunnel) : (str, object)
        The 0MQ url that has been forwarded, and the tunnel object
    """
    lport = select_random_ports(1)[0]
    transport, addr = addr.split('://')
    ip, rport = addr.split(':')
    rport = int(rport)
    if paramiko is None:
        paramiko = sys.platform == 'win32'
    if paramiko:
        tunnelf = paramiko_tunnel
    else:
        tunnelf = openssh_tunnel
    tunnel = tunnelf(lport, rport, server, remoteip=ip, keyfile=keyfile, password=password, timeout=timeout)
    return ('tcp://127.0.0.1:%i' % lport, tunnel)
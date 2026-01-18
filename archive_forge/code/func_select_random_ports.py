import atexit
import os
import re
import signal
import socket
import sys
import warnings
from getpass import getpass, getuser
from multiprocessing import Process
def select_random_ports(n):
    """Select and return n random ports that are available."""
    ports = []
    sockets = []
    for i in range(n):
        sock = socket.socket()
        sock.bind(('', 0))
        ports.append(sock.getsockname()[1])
        sockets.append(sock)
    for sock in sockets:
        sock.close()
    return ports
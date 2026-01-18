import errno
import os
import sys
import time
import traceback
import types
import urllib.parse
import warnings
import eventlet
from eventlet import greenio
from eventlet import support
from eventlet.corolocal import local
from eventlet.green import BaseHTTPServer
from eventlet.green import socket
def addr_to_host_port(addr):
    host = 'unix'
    port = ''
    if isinstance(addr, tuple):
        host = addr[0]
        port = addr[1]
    return (host, port)
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
def send_hundred_continue_response(self):
    if self.headers_sent:
        return
    towrite = []
    towrite.append(self.wfile_line)
    if self.hundred_continue_headers is not None:
        for header in self.hundred_continue_headers:
            towrite.append(('%s: %s\r\n' % header).encode())
    towrite.append(b'\r\n')
    self.wfile.writelines(towrite)
    self.wfile.flush()
    self.chunk_length = -1
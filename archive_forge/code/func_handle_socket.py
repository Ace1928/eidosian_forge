import errno
import fcntl
import os
from oslo_log import log as logging
import select
import signal
import socket
import ssl
import struct
import sys
import termios
import time
import tty
from urllib import parse as urlparse
import websocket
from zunclient.common.apiclient import exceptions as acexceptions
from zunclient.common.websocketclient import exceptions
def handle_socket(self, event):
    if event in (select.POLLHUP, select.POLLNVAL):
        self.poll.unregister(self.fileno())
        self.quit = True
    data = self.recv()
    if not data:
        self.poll.unregister(self.fileno())
        self.quit = True
        return
    if isinstance(data, bytes):
        data = data.decode('utf-8')
    sys.stdout.write(data)
    sys.stdout.flush()
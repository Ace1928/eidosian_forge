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
def handle_stdin(self, event):
    if event in (select.POLLHUP, select.POLLNVAL):
        LOG.debug('event %d on stdin', event)
        LOG.debug('eof on stdin')
        self.poll.unregister(sys.stdin)
        self.quit = True
    data = os.read(sys.stdin.fileno(), 1024)
    if not data:
        return
    if self.start_of_line and data == self.escape:
        self.read_escape = True
        return
    if self.read_escape and data == '.':
        LOG.debug('exit by local escape code')
        raise exceptions.UserExit()
    elif self.read_escape:
        self.read_escape = False
        self.send(self.escape)
    self.send(data)
    if data == '\r':
        self.start_of_line = True
    else:
        self.start_of_line = False
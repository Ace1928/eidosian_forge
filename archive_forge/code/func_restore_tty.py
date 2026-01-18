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
def restore_tty(self):
    if os.isatty(sys.stdin.fileno()):
        LOG.debug('restoring tty configuration')
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
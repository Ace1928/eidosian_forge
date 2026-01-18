import sys
import struct
import ssl
from base64 import b64encode
from hashlib import sha1
import logging
from socket import error as SocketError
import errno
import threading
from socketserver import ThreadingMixIn, TCPServer, StreamRequestHandler
from websocket_server.thread import WebsocketServerThread
def deny_new_connections(self, status=CLOSE_STATUS_NORMAL, reason=DEFAULT_CLOSE_REASON):
    self._deny_new_connections(status, reason)
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
def read_http_headers(self):
    headers = {}
    http_get = self.rfile.readline().decode().strip()
    assert http_get.upper().startswith('GET')
    while True:
        header = self.rfile.readline().decode().strip()
        if not header:
            break
        head, value = header.split(':', 1)
        headers[head.lower().strip()] = value.strip()
    return headers
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
@classmethod
def calculate_response_key(cls, key):
    GUID = '258EAFA5-E914-47DA-95CA-C5AB0DC85B11'
    hash = sha1(key.encode() + GUID.encode())
    response_key = b64encode(hash.digest()).strip()
    return response_key.decode('ASCII')
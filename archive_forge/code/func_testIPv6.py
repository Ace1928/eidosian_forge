import os
import os.path
import socket
import unittest
from base64 import decodebytes as base64decode
import websocket as ws
from websocket._handshake import _create_sec_websocket_key
from websocket._handshake import _validate as _validate_header
from websocket._http import read_headers
from websocket._utils import validate_utf8
def testIPv6(self):
    websock2 = ws.WebSocket()
    self.assertRaises(ValueError, websock2.connect, '2001:4860:4860::8888')
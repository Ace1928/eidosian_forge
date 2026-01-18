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
def testRecv(self):
    sock = ws.WebSocket()
    s = sock.sock = SockMock()
    something = b'\x81\x8fabcd\x82\xe3\xf0\x87\xe3\xf1\x80\xe5\xca\x81\xe2\xc5\x82\xe3\xcc'
    s.add_packet(something)
    data = sock.recv()
    self.assertEqual(data, 'こんにちは')
    s.add_packet(b'\x81\x85abcd)\x07\x0f\x08\x0e')
    data = sock.recv()
    self.assertEqual(data, 'Hello')
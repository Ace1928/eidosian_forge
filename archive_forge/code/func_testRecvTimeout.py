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
def testRecvTimeout(self):
    sock = ws.WebSocket()
    s = sock.sock = SockMock()
    s.add_packet(b'\x81')
    s.add_packet(socket.timeout())
    s.add_packet(b'\x8dabcd)\x07\x0f\x08\x0e')
    s.add_packet(socket.timeout())
    s.add_packet(b'NC3\x0e\x10\x0f\x00@')
    with self.assertRaises(ws.WebSocketTimeoutException):
        sock.recv()
    with self.assertRaises(ws.WebSocketTimeoutException):
        sock.recv()
    data = sock.recv()
    self.assertEqual(data, 'Hello, World!')
    with self.assertRaises(ws.WebSocketConnectionClosedException):
        sock.recv()
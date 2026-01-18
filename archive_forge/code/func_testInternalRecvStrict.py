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
def testInternalRecvStrict(self):
    sock = ws.WebSocket()
    s = sock.sock = SockMock()
    s.add_packet(b'foo')
    s.add_packet(socket.timeout())
    s.add_packet(b'bar')
    s.add_packet(b'baz')
    with self.assertRaises(ws.WebSocketTimeoutException):
        sock.frame_buffer.recv_strict(9)
    data = sock.frame_buffer.recv_strict(9)
    self.assertEqual(data, b'foobarbaz')
    with self.assertRaises(ws.WebSocketConnectionClosedException):
        sock.frame_buffer.recv_strict(1)
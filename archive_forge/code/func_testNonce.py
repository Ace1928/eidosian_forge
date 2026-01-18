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
def testNonce(self):
    """WebSocket key should be a random 16-byte nonce."""
    key = _create_sec_websocket_key()
    nonce = base64decode(key.encode('utf-8'))
    self.assertEqual(16, len(nonce))
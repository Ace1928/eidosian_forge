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
def testWSKey(self):
    key = _create_sec_websocket_key()
    self.assertTrue(key != 24)
    self.assertTrue('¥n' not in key)
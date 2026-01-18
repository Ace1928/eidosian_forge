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
def testUtf8Validator(self):
    state = validate_utf8(b'\xf0\x90\x80\x80')
    self.assertEqual(state, True)
    state = validate_utf8(b'\xce\xba\xe1\xbd\xb9\xcf\x83\xce\xbc\xce\xb5\xed\xa0\x80edited')
    self.assertEqual(state, False)
    state = validate_utf8(b'')
    self.assertEqual(state, True)
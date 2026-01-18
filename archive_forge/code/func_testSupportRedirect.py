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
@unittest.skipUnless(TEST_WITH_INTERNET, 'Internet-requiring tests are disabled')
def testSupportRedirect(self):
    s = ws.WebSocket()
    self.assertRaises(ws._exceptions.WebSocketBadStatusException, s.connect, 'ws://google.com/')
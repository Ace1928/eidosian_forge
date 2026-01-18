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
@unittest.skipUnless(TEST_WITH_LOCAL_SERVER, 'Tests using local websocket server are disabled')
def testPingPong(self):
    s = ws.create_connection(f'ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}')
    self.assertNotEqual(s, None)
    s.ping('Hello')
    s.pong('Hi')
    s.close()
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
def testSecureWebSocket(self):
    import ssl
    s = ws.create_connection('wss://api.bitfinex.com/ws/2')
    self.assertNotEqual(s, None)
    self.assertTrue(isinstance(s.sock, ssl.SSLSocket))
    self.assertEqual(s.getstatus(), 101)
    self.assertNotEqual(s.getheaders(), None)
    s.settimeout(10)
    self.assertEqual(s.gettimeout(), 10)
    self.assertEqual(s.getsubprotocol(), None)
    s.abort()
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
def testManualHeaders(self):
    websock3 = ws.WebSocket(sslopt={'ca_certs': ssl.get_default_verify_paths().cafile, 'ca_cert_path': ssl.get_default_verify_paths().capath})
    self.assertRaises(ws._exceptions.WebSocketBadStatusException, websock3.connect, 'wss://api.bitfinex.com/ws/2', cookie='chocolate', origin='testing_websockets.com', host='echo.websocket.events/websocket-client-test', subprotocols=['testproto'], connection='Upgrade', header={'CustomHeader1': '123', 'Cookie': 'TestValue', 'Sec-WebSocket-Key': 'k9kFAUWNAMmf5OEMfTlOEA==', 'Sec-WebSocket-Protocol': 'newprotocol'})
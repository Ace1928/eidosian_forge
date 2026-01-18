import os
import os.path
import ssl
import threading
import unittest
import websocket as ws
@unittest.skipUnless(TEST_WITH_INTERNET, 'Internet-requiring tests are disabled')
def testBadPingTimeout(self):
    """A WebSocketApp handling of negative ping_timeout"""
    app = ws.WebSocketApp('wss://api-pub.bitfinex.com/ws/1')
    self.assertRaises(ws.WebSocketException, app.run_forever, ping_timeout=-3, sslopt={'cert_reqs': ssl.CERT_NONE})
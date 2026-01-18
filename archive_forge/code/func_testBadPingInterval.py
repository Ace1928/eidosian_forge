import os
import os.path
import ssl
import threading
import unittest
import websocket as ws
@unittest.skipUnless(TEST_WITH_INTERNET, 'Internet-requiring tests are disabled')
def testBadPingInterval(self):
    """A WebSocketApp handling of negative ping_interval"""
    app = ws.WebSocketApp('wss://api-pub.bitfinex.com/ws/1')
    self.assertRaises(ws.WebSocketException, app.run_forever, ping_interval=-5, sslopt={'cert_reqs': ssl.CERT_NONE})
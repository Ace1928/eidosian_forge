import os
import os.path
import ssl
import threading
import unittest
import websocket as ws
@unittest.skipUnless(TEST_WITH_INTERNET, 'Internet-requiring tests are disabled')
def testInvalidPingIntervalPingTimeout(self):
    """Test exception handling if ping_interval < ping_timeout"""

    def on_ping(app, msg):
        print('Got a ping!')
        app.close()

    def on_pong(app, msg):
        print('Got a pong! No need to respond')
        app.close()
    app = ws.WebSocketApp('wss://api-pub.bitfinex.com/ws/1', on_ping=on_ping, on_pong=on_pong)
    self.assertRaises(ws.WebSocketException, app.run_forever, ping_interval=1, ping_timeout=2, sslopt={'cert_reqs': ssl.CERT_NONE})
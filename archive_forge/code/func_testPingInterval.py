import os
import os.path
import ssl
import threading
import unittest
import websocket as ws
@unittest.skipUnless(TEST_WITH_INTERNET, 'Internet-requiring tests are disabled')
def testPingInterval(self):
    """Test WebSocketApp proper ping functionality"""

    def on_ping(app, msg):
        print('Got a ping!')
        app.close()

    def on_pong(app, msg):
        print('Got a pong! No need to respond')
        app.close()
    app = ws.WebSocketApp('wss://api-pub.bitfinex.com/ws/1', on_ping=on_ping, on_pong=on_pong)
    app.run_forever(ping_interval=2, ping_timeout=1, sslopt={'cert_reqs': ssl.CERT_NONE})
import os
import os.path
import ssl
import threading
import unittest
import websocket as ws
@unittest.skipUnless(TEST_WITH_INTERNET, 'Internet-requiring tests are disabled')
def testOpcodeClose(self):
    """Test WebSocketApp close opcode"""
    app = ws.WebSocketApp('wss://tsock.us1.twilio.com/v3/wsconnect')
    app.run_forever(ping_interval=2, ping_timeout=1, ping_payload='Ping payload')
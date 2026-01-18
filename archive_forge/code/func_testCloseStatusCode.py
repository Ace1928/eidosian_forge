import os
import os.path
import ssl
import threading
import unittest
import websocket as ws
@unittest.skipUnless(TEST_WITH_INTERNET, 'Internet-requiring tests are disabled')
def testCloseStatusCode(self):
    """Test extraction of close frame status code and close reason in WebSocketApp"""

    def on_close(wsapp, close_status_code, close_msg):
        print('on_close reached')
    app = ws.WebSocketApp('wss://tsock.us1.twilio.com/v3/wsconnect', on_close=on_close)
    closeframe = ws.ABNF(opcode=ws.ABNF.OPCODE_CLOSE, data=b'\x03\xe8no-init-from-client')
    self.assertEqual([1000, 'no-init-from-client'], app._get_close_args(closeframe))
    closeframe = ws.ABNF(opcode=ws.ABNF.OPCODE_CLOSE, data=b'')
    self.assertEqual([None, None], app._get_close_args(closeframe))
    app2 = ws.WebSocketApp('wss://tsock.us1.twilio.com/v3/wsconnect')
    closeframe = ws.ABNF(opcode=ws.ABNF.OPCODE_CLOSE, data=b'')
    self.assertEqual([None, None], app2._get_close_args(closeframe))
    self.assertRaises(ws.WebSocketConnectionClosedException, app.send, data='test if connection is closed')
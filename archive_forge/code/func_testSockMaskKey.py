import os
import os.path
import ssl
import threading
import unittest
import websocket as ws
@unittest.skipUnless(TEST_WITH_INTERNET, 'Internet-requiring tests are disabled')
def testSockMaskKey(self):
    """A WebSocketApp should forward the received mask_key function down
        to the actual socket.
        """

    def my_mask_key_func():
        return '\x00\x00\x00\x00'
    app = ws.WebSocketApp('wss://api-pub.bitfinex.com/ws/1', get_mask_key=my_mask_key_func)
    self.assertEqual(id(app.get_mask_key), id(my_mask_key_func))
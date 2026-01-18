import os
import os.path
import ssl
import threading
import unittest
import websocket as ws
@unittest.skipUnless(TEST_WITH_LOCAL_SERVER, 'Tests using local websocket server are disabled')
def testReconnect(self):
    """Test reconnect"""
    pong_count = 0
    exc = None

    def on_error(app, err):
        nonlocal exc
        exc = err

    def on_pong(app, msg):
        nonlocal pong_count
        pong_count += 1
        if pong_count == 1:
            app.sock.shutdown()
        if pong_count >= 2:
            app.close()
    app = ws.WebSocketApp(f'ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}', on_pong=on_pong, on_error=on_error)
    app.run_forever(ping_interval=2, ping_timeout=1, reconnect=3)
    self.assertEqual(pong_count, 2)
    self.assertIsInstance(exc, ws.WebSocketTimeoutException)
    self.assertEqual(str(exc), 'ping/pong timed out')
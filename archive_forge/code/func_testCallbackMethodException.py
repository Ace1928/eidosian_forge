import os
import os.path
import ssl
import threading
import unittest
import websocket as ws
@unittest.skipUnless(TEST_WITH_LOCAL_SERVER, 'Tests using local websocket server are disabled')
def testCallbackMethodException(self):
    """Test callback method exception handling"""

    class Callbacks:

        def __init__(self):
            self.exc = None
            self.passed_app = None
            self.app = ws.WebSocketApp(f'ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}', on_open=self.on_open, on_error=self.on_error, on_pong=self.on_pong)
            self.app.run_forever(ping_interval=2, ping_timeout=1)

        def on_open(self, app):
            raise RuntimeError('Callback failed')

        def on_error(self, app, err):
            self.passed_app = app
            self.exc = err

        def on_pong(self, app, msg):
            app.close()
    callbacks = Callbacks()
    self.assertEqual(callbacks.passed_app, callbacks.app)
    self.assertIsInstance(callbacks.exc, RuntimeError)
    self.assertEqual(str(callbacks.exc), 'Callback failed')
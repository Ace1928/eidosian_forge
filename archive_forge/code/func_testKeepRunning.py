import os
import os.path
import ssl
import threading
import unittest
import websocket as ws
@unittest.skipUnless(TEST_WITH_LOCAL_SERVER, 'Tests using local websocket server are disabled')
def testKeepRunning(self):
    """A WebSocketApp should keep running as long as its self.keep_running
        is not False (in the boolean context).
        """

    def on_open(self, *args, **kwargs):
        """Set the keep_running flag for later inspection and immediately
            close the connection.
            """
        self.send('hello!')
        WebSocketAppTest.keep_running_open = self.keep_running
        self.keep_running = False

    def on_message(wsapp, message):
        print(message)
        self.close()

    def on_close(self, *args, **kwargs):
        """Set the keep_running flag for the test to use."""
        WebSocketAppTest.keep_running_close = self.keep_running
    app = ws.WebSocketApp(f'ws://127.0.0.1:{LOCAL_WS_SERVER_PORT}', on_open=on_open, on_close=on_close, on_message=on_message)
    app.run_forever()
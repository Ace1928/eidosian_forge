import os
import os.path
import ssl
import threading
import unittest
import websocket as ws
def on_ping(app, msg):
    print('Got a ping!')
    app.close()
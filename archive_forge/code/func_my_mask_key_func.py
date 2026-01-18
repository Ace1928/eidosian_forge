import os
import os.path
import ssl
import threading
import unittest
import websocket as ws
def my_mask_key_func():
    return '\x00\x00\x00\x00'
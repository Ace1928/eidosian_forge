import os
import unittest
from websocket._url import (
def testMatchAll(self):
    self.assertTrue(_is_no_proxy_host('any.websocket.org', ['*']))
    self.assertTrue(_is_no_proxy_host('192.168.0.1', ['*']))
    self.assertTrue(_is_no_proxy_host('any.websocket.org', ['other.websocket.org', '*']))
    os.environ['no_proxy'] = '*'
    self.assertTrue(_is_no_proxy_host('any.websocket.org', None))
    self.assertTrue(_is_no_proxy_host('192.168.0.1', None))
    os.environ['no_proxy'] = 'other.websocket.org, *'
    self.assertTrue(_is_no_proxy_host('any.websocket.org', None))
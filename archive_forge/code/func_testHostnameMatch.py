import os
import unittest
from websocket._url import (
def testHostnameMatch(self):
    self.assertTrue(_is_no_proxy_host('my.websocket.org', ['my.websocket.org']))
    self.assertTrue(_is_no_proxy_host('my.websocket.org', ['other.websocket.org', 'my.websocket.org']))
    self.assertFalse(_is_no_proxy_host('my.websocket.org', ['other.websocket.org']))
    os.environ['no_proxy'] = 'my.websocket.org'
    self.assertTrue(_is_no_proxy_host('my.websocket.org', None))
    self.assertFalse(_is_no_proxy_host('other.websocket.org', None))
    os.environ['no_proxy'] = 'other.websocket.org, my.websocket.org'
    self.assertTrue(_is_no_proxy_host('my.websocket.org', None))
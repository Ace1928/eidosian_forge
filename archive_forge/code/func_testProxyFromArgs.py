import os
import unittest
from websocket._url import (
def testProxyFromArgs(self):
    self.assertEqual(get_proxy_info('echo.websocket.events', False, proxy_host='localhost'), ('localhost', 0, None))
    self.assertEqual(get_proxy_info('echo.websocket.events', False, proxy_host='localhost', proxy_port=3128), ('localhost', 3128, None))
    self.assertEqual(get_proxy_info('echo.websocket.events', True, proxy_host='localhost'), ('localhost', 0, None))
    self.assertEqual(get_proxy_info('echo.websocket.events', True, proxy_host='localhost', proxy_port=3128), ('localhost', 3128, None))
    self.assertEqual(get_proxy_info('echo.websocket.events', False, proxy_host='localhost', proxy_auth=('a', 'b')), ('localhost', 0, ('a', 'b')))
    self.assertEqual(get_proxy_info('echo.websocket.events', False, proxy_host='localhost', proxy_port=3128, proxy_auth=('a', 'b')), ('localhost', 3128, ('a', 'b')))
    self.assertEqual(get_proxy_info('echo.websocket.events', True, proxy_host='localhost', proxy_auth=('a', 'b')), ('localhost', 0, ('a', 'b')))
    self.assertEqual(get_proxy_info('echo.websocket.events', True, proxy_host='localhost', proxy_port=3128, proxy_auth=('a', 'b')), ('localhost', 3128, ('a', 'b')))
    self.assertEqual(get_proxy_info('echo.websocket.events', True, proxy_host='localhost', proxy_port=3128, no_proxy=['example.com'], proxy_auth=('a', 'b')), ('localhost', 3128, ('a', 'b')))
    self.assertEqual(get_proxy_info('echo.websocket.events', True, proxy_host='localhost', proxy_port=3128, no_proxy=['echo.websocket.events'], proxy_auth=('a', 'b')), (None, 0, None))
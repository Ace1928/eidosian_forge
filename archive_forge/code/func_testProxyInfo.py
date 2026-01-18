import os
import os.path
import socket
import ssl
import unittest
import websocket
import websocket as ws
from websocket._http import (
def testProxyInfo(self):
    self.assertEqual(proxy_info(http_proxy_host='127.0.0.1', http_proxy_port='8080', proxy_type='http').proxy_protocol, 'http')
    self.assertRaises(ProxyError, proxy_info, http_proxy_host='127.0.0.1', http_proxy_port='8080', proxy_type='badval')
    self.assertEqual(proxy_info(http_proxy_host='example.com', http_proxy_port='8080', proxy_type='http').proxy_host, 'example.com')
    self.assertEqual(proxy_info(http_proxy_host='127.0.0.1', http_proxy_port='8080', proxy_type='http').proxy_port, '8080')
    self.assertEqual(proxy_info(http_proxy_host='127.0.0.1', http_proxy_port='8080', proxy_type='http').auth, None)
    self.assertEqual(proxy_info(http_proxy_host='127.0.0.1', http_proxy_port='8080', proxy_type='http', http_proxy_auth=('my_username123', 'my_pass321')).auth[0], 'my_username123')
    self.assertEqual(proxy_info(http_proxy_host='127.0.0.1', http_proxy_port='8080', proxy_type='http', http_proxy_auth=('my_username123', 'my_pass321')).auth[1], 'my_pass321')
import os
import os.path
import socket
import ssl
import unittest
import websocket
import websocket as ws
from websocket._http import (
@unittest.skipUnless(TEST_WITH_INTERNET, 'Internet-requiring tests are disabled')
def testSSLopt(self):
    ssloptions = {'check_hostname': False, 'server_hostname': 'ServerName', 'ssl_version': ssl.PROTOCOL_TLS_CLIENT, 'ciphers': 'TLS_AES_256_GCM_SHA384:TLS_CHACHA20_POLY1305_SHA256:                        TLS_AES_128_GCM_SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:                        ECDHE-RSA-AES256-GCM-SHA384:DHE-RSA-AES256-GCM-SHA384:                        ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:                        DHE-RSA-CHACHA20-POLY1305:ECDHE-ECDSA-AES128-GCM-SHA256:                        ECDHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES128-GCM-SHA256:                        ECDHE-ECDSA-AES256-SHA384:ECDHE-RSA-AES256-SHA384:                        DHE-RSA-AES256-SHA256:ECDHE-ECDSA-AES128-SHA256:                        ECDHE-RSA-AES128-SHA256:DHE-RSA-AES128-SHA256:                        ECDHE-ECDSA-AES256-SHA:ECDHE-RSA-AES256-SHA', 'ecdh_curve': 'prime256v1'}
    ws_ssl1 = websocket.WebSocket(sslopt=ssloptions)
    ws_ssl1.connect('wss://api.bitfinex.com/ws/2')
    ws_ssl1.send('Hello')
    ws_ssl1.close()
    ws_ssl2 = websocket.WebSocket(sslopt={'check_hostname': True})
    ws_ssl2.connect('wss://api.bitfinex.com/ws/2')
    ws_ssl2.close
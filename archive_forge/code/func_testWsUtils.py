import os
import os.path
import socket
import unittest
from base64 import decodebytes as base64decode
import websocket as ws
from websocket._handshake import _create_sec_websocket_key
from websocket._handshake import _validate as _validate_header
from websocket._http import read_headers
from websocket._utils import validate_utf8
def testWsUtils(self):
    key = 'c6b8hTg4EeGb2gQMztV1/g=='
    required_header = {'upgrade': 'websocket', 'connection': 'upgrade', 'sec-websocket-accept': 'Kxep+hNu9n51529fGidYu7a3wO0='}
    self.assertEqual(_validate_header(required_header, key, None), (True, None))
    header = required_header.copy()
    header['upgrade'] = 'http'
    self.assertEqual(_validate_header(header, key, None), (False, None))
    del header['upgrade']
    self.assertEqual(_validate_header(header, key, None), (False, None))
    header = required_header.copy()
    header['connection'] = 'something'
    self.assertEqual(_validate_header(header, key, None), (False, None))
    del header['connection']
    self.assertEqual(_validate_header(header, key, None), (False, None))
    header = required_header.copy()
    header['sec-websocket-accept'] = 'something'
    self.assertEqual(_validate_header(header, key, None), (False, None))
    del header['sec-websocket-accept']
    self.assertEqual(_validate_header(header, key, None), (False, None))
    header = required_header.copy()
    header['sec-websocket-protocol'] = 'sub1'
    self.assertEqual(_validate_header(header, key, ['sub1', 'sub2']), (True, 'sub1'))
    self.assertEqual(_validate_header(header, key, ['sub2', 'sub3']), (False, None))
    header = required_header.copy()
    header['sec-websocket-protocol'] = 'sUb1'
    self.assertEqual(_validate_header(header, key, ['Sub1', 'suB2']), (True, 'sub1'))
    header = required_header.copy()
    self.assertEqual(_validate_header(header, key, ['Sub1', 'suB2']), (False, None))
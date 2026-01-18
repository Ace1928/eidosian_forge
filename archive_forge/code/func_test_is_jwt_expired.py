import sys
import json
import time
import base64
import unittest
from unittest import mock
import libcloud.common.gig_g8
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, StorageVolume
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gig_g8 import G8Network, G8NodeDriver, G8PortForward
def test_is_jwt_expired(self):
    data = {'azp': 'example', 'exp': int(time.time()), 'iss': 'itsyouonline', 'refresh_token': 'xxxxxxx', 'scope': ['user:admin'], 'username': 'example'}

    def contruct_jwt(data):
        jsondata = json.dumps(data).encode()
        return 'header.{}.signature'.format(base64.encodebytes(jsondata).decode())
    libcloud.common.gig_g8.is_jwt_expired = original_is_jwt_expired
    self.assertTrue(libcloud.common.gig_g8.is_jwt_expired(contruct_jwt(data)))
    data['exp'] = int(time.time()) + 300
    self.assertFalse(libcloud.common.gig_g8.is_jwt_expired(contruct_jwt(data)))
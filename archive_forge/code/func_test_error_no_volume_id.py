import json
import os
from unittest import mock
import requests
from os_brick import exception
from os_brick.initiator.connectors import scaleio
from os_brick.tests.initiator import test_connector
def test_error_no_volume_id(self):
    """Faile to connect with no volume id"""
    self.fake_connection_properties['scaleIO_volume_id'] = None
    self.mock_calls[self.get_volume_api] = self.MockHTTPSResponse('null', 200)
    self.assertRaises(exception.BrickException, self.test_connect_volume)
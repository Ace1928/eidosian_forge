import json
import os
from unittest import mock
import requests
from os_brick import exception
from os_brick.initiator.connectors import scaleio
from os_brick.tests.initiator import test_connector
def test_error_id(self):
    """Fail to connect with bad volume name"""
    self.fake_connection_properties['scaleIO_volume_id'] = 'bad_id'
    self.mock_calls[self.get_volume_api] = self.MockHTTPSResponse(dict(errorCode='404', message='Test volume not found'), 404)
    self.assertRaises(exception.BrickException, self.test_connect_volume)
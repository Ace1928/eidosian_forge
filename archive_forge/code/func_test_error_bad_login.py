import json
import os
from unittest import mock
import requests
from os_brick import exception
from os_brick.initiator.connectors import scaleio
from os_brick.tests.initiator import test_connector
def test_error_bad_login(self):
    """Fail to connect with bad authentication"""
    self.mock_calls[self.get_volume_api] = self.MockHTTPSResponse('null', 401)
    self.mock_calls['login'] = self.MockHTTPSResponse('null', 401)
    self.mock_calls[self.action_format.format('addMappedSdc')] = self.MockHTTPSResponse(dict(errorCode=401, message='bad login'), 401)
    self.assertRaises(exception.BrickException, self.test_connect_volume)
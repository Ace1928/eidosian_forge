import json
import os
from unittest import mock
import requests
from os_brick import exception
from os_brick.initiator.connectors import scaleio
from os_brick.tests.initiator import test_connector
@mock.patch('os_brick.utils._time_sleep')
def test_error_path_not_found(self, sleep_mock):
    """Timeout waiting for volume to map to local file system"""
    self.mock_object(os, 'listdir', return_value=['emc-vol-no-volume'])
    self.assertRaises(exception.BrickException, self.test_connect_volume)
    self.assertTrue(sleep_mock.called)
import json
import os
from unittest import mock
import requests
from os_brick import exception
from os_brick.initiator.connectors import scaleio
from os_brick.tests.initiator import test_connector
def test_error_disconnect_volume(self):
    """Fail to disconnect with REST API failure"""
    self.mock_calls[self.action_format.format('removeMappedSdc')] = self.MockHTTPSResponse(dict(errorCode=self.connector.VOLUME_ALREADY_MAPPED_ERROR, message='Test error map volume'), 500)
    self.assertRaises(exception.BrickException, self.test_disconnect_volume)
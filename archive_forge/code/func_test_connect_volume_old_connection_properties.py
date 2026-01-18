import json
import os
from unittest import mock
import requests
from os_brick import exception
from os_brick.initiator.connectors import scaleio
from os_brick.tests.initiator import test_connector
def test_connect_volume_old_connection_properties(self):
    """Successful connect to volume"""
    connection_properties = {'hostIP': test_connector.MY_IP, 'serverIP': test_connector.MY_IP, 'scaleIO_volname': self.vol['name'], 'scaleIO_volume_id': self.vol['provider_id'], 'serverPort': 443, 'serverUsername': 'test', 'serverPassword': 'fake', 'serverToken': 'fake_token', 'iopsLimit': None, 'bandwidthLimit': None}
    self.connector.connect_volume(connection_properties)
    self.get_guid_mock.assert_called_once_with(self.connector.GET_GUID_OP_CODE)
    self.get_password_mock.assert_not_called()
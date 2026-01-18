import socket
from unittest import mock
from octavia_lib.api.drivers import driver_lib
from octavia_lib.api.drivers import exceptions as driver_exceptions
from octavia_lib.common import constants
from octavia_lib.tests.unit import base
@mock.patch('octavia_lib.api.drivers.data_models.Listener.from_dict')
def test_get_listener(self, mock_from_dict):
    self._test_get_object(self.driver_lib.get_listener, constants.LISTENERS, mock_from_dict)
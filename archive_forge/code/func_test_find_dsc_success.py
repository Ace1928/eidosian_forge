import glob
import http.client
import queue
from unittest import mock
from unittest.mock import mock_open
from os_brick import exception
from os_brick.initiator.connectors import lightos
from os_brick.initiator import linuxscsi
from os_brick.privileged import lightos as priv_lightos
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch.object(lightos.http.client.HTTPConnection, 'request', return_value=None)
@mock.patch.object(lightos.http.client.HTTPConnection, 'getresponse', return_value=get_http_response_mock(http.client.OK))
def test_find_dsc_success(self, mocked_connection, mocked_response):
    mocked_connection.request.return_value = None
    mocked_response.getresponse.return_value = get_http_response_mock(http.client.OK)
    self.assertEqual(self.connector.find_dsc(), 'found')
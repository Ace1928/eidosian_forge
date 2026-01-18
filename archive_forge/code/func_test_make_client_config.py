from unittest import mock
from taskflow import test
from taskflow.utils import kazoo_utils
@mock.patch('kazoo.client.KazooClient')
def test_make_client_config(self, mock_kazoo_client):
    conf = {}
    expected = {'hosts': 'localhost:2181', 'logger': mock.ANY, 'read_only': False, 'randomize_hosts': False, 'keyfile': None, 'keyfile_password': None, 'certfile': None, 'use_ssl': False, 'verify_certs': True}
    kazoo_utils.make_client(conf)
    mock_kazoo_client.assert_called_once_with(**expected)
    mock_kazoo_client.reset_mock()
    conf = {'use_ssl': 'True', 'verify_certs': 'False'}
    expected = {'hosts': 'localhost:2181', 'logger': mock.ANY, 'read_only': False, 'randomize_hosts': False, 'keyfile': None, 'keyfile_password': None, 'certfile': None, 'use_ssl': True, 'verify_certs': False}
    kazoo_utils.make_client(conf)
    mock_kazoo_client.assert_called_once_with(**expected)
    mock_kazoo_client.reset_mock()
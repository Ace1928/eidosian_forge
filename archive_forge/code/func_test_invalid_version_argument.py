import testtools
from unittest import mock
from magnumclient import client
@mock.patch('magnumclient.v1.client.Client')
def test_invalid_version_argument(self, mock_magnum_client):
    self.assertRaises(ValueError, client.Client, version='2', magnum_url='http://myurl/')
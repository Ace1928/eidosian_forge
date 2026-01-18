from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import client
from manilaclient import exceptions
from manilaclient.tests.unit import utils
import manilaclient.v1.client
import manilaclient.v2.client
@ddt.data(None, '', '3', 'v1', 'v2', 'v1.0', 'v2.0')
def test_init_client_with_unsupported_version(self, v):
    self.assertRaises(exceptions.UnsupportedVersion, client.Client, v)
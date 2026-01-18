from unittest import mock
import ddt
from oslo_utils import uuidutils
import manilaclient
from manilaclient import exceptions
from manilaclient.tests.unit import utils
from manilaclient.v2 import client
def test_auth_via_token_invalid(self):
    self.assertRaises(exceptions.ClientException, client.Client, api_version=manilaclient.API_MAX_VERSION, input_auth_token='token')
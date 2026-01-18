import argparse
import io
import re
import sys
from unittest import mock
import ddt
import fixtures
from keystoneauth1 import fixture
import requests_mock
from testtools import matchers
from novaclient import api_versions
import novaclient.client
from novaclient import exceptions
import novaclient.shell
from novaclient.tests.unit import fake_actions_module
from novaclient.tests.unit import utils
def test_microversion_with_v2_and_v2_1_server(self):
    self.make_env()
    self.mock_server_version_range.return_value = (api_versions.APIVersion('2.1'), api_versions.APIVersion('2.3'))
    novaclient.API_MAX_VERSION = api_versions.APIVersion('2.100')
    novaclient.API_MIN_VERSION = api_versions.APIVersion('2.1')
    self.shell('--os-compute-api-version 2 list')
    client_args = self.mock_client.call_args_list[1][0]
    self.assertEqual(api_versions.APIVersion('2.0'), client_args[0])
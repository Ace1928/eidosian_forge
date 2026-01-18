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
def test_default_endpoint_type(self):
    self.make_env()
    self.shell('list')
    client_kwargs = self.mock_client.call_args_list[0][1]
    self.assertEqual(client_kwargs['endpoint_type'], 'publicURL')
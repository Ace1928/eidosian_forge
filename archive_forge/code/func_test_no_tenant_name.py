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
def test_no_tenant_name(self):
    required = self._msg_no_tenant_project
    self.make_env(exclude='OS_TENANT_NAME')
    try:
        self.shell('list')
    except exceptions.CommandError as message:
        self.assertEqual(required, message.args[0])
    else:
        self.fail('CommandError not raised')
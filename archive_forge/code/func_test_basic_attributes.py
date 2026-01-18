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
@ddt.data((None, 'project_domain_id', FAKE_ENV['OS_PROJECT_DOMAIN_ID']), ('OS_PROJECT_DOMAIN_ID', 'project_domain_id', ''), (None, 'project_domain_name', FAKE_ENV['OS_PROJECT_DOMAIN_NAME']), ('OS_PROJECT_DOMAIN_NAME', 'project_domain_name', ''), (None, 'user_domain_id', FAKE_ENV['OS_USER_DOMAIN_ID']), ('OS_USER_DOMAIN_ID', 'user_domain_id', ''), (None, 'user_domain_name', FAKE_ENV['OS_USER_DOMAIN_NAME']), ('OS_USER_DOMAIN_NAME', 'user_domain_name', ''))
@ddt.unpack
def test_basic_attributes(self, exclude, client_arg, env_var):
    self.make_env(exclude=exclude, fake_env=FAKE_ENV)
    self.shell('list')
    client_kwargs = self.mock_client.call_args_list[0][1]
    self.assertEqual(env_var, client_kwargs[client_arg])
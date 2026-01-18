import io
import re
import sys
from unittest import mock
import ddt
import fixtures
from tempest.lib.cli import output_parser
from testtools import matchers
import manilaclient
from manilaclient.common import cliutils
from manilaclient.common import constants
from manilaclient import exceptions
from manilaclient import shell
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
@ddt.data({'env_vars': {'OS_MANILA_BYPASS_URL': 'http://foo.url', 'OS_TOKEN': 'foo_token'}, 'kwargs': {'--os-token': 'bar_token', '--bypass-url': 'http://bar.url'}, 'expected': {'input_auth_token': 'bar_token', 'service_catalog_url': 'http://bar.url'}}, {'env_vars': {'OS_MANILA_BYPASS_URL': 'http://foo.url', 'OS_TOKEN': 'foo_token'}, 'kwargs': {}, 'expected': {'input_auth_token': 'foo_token', 'service_catalog_url': 'http://foo.url'}}, {'env_vars': {}, 'kwargs': {'--os-token': 'bar_token', '--bypass-url': 'http://bar.url'}, 'expected': {'input_auth_token': 'bar_token', 'service_catalog_url': 'http://bar.url'}}, {'env_vars': {'MANILACLIENT_BYPASS_URL': 'http://foo.url', 'OS_TOKEN': 'foo_token'}, 'kwargs': {}, 'expected': {'input_auth_token': 'foo_token', 'service_catalog_url': 'http://foo.url'}}, {'env_vars': {'OS_TOKEN': 'foo_token'}, 'kwargs': {'--bypass-url': 'http://bar.url'}, 'expected': {'input_auth_token': 'foo_token', 'service_catalog_url': 'http://bar.url'}}, {'env_vars': {'MANILACLIENT_BYPASS_URL': 'http://foo.url', 'OS_MANILA_BYPASS_URL': 'http://bar.url', 'OS_TOKEN': 'foo_token'}, 'kwargs': {'--os-token': 'bar_token'}, 'expected': {'input_auth_token': 'bar_token', 'service_catalog_url': 'http://bar.url'}})
@ddt.unpack
def test_main_success_with_token(self, env_vars, kwargs, expected):
    self.set_env_vars(env_vars)
    with mock.patch.object(shell, 'client') as mock_client:
        cmd = ''
        for k, v in kwargs.items():
            cmd += '%s=%s ' % (k, v)
        cmd += 'list'
        self.shell(cmd)
        mock_client.Client.assert_called_with(manilaclient.API_MAX_VERSION, username='', password='', project_name='', auth_url='', insecure=False, region_name='', tenant_id='', endpoint_type='publicURL', extensions=mock.ANY, service_type=constants.V2_SERVICE_TYPE, service_name='', retries=0, http_log_debug=False, cacert=None, use_keyring=False, force_new_token=False, user_id='', user_domain_id='', user_domain_name='', project_domain_id='', project_domain_name='', cert=None, input_auth_token=expected['input_auth_token'], service_catalog_url=expected['service_catalog_url'])
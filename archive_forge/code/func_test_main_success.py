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
@ddt.data(None, 'foo_key')
def test_main_success(self, os_key):
    env_vars = {'OS_AUTH_URL': 'http://foo.bar', 'OS_USERNAME': 'foo_username', 'OS_USER_ID': 'foo_user_id', 'OS_PASSWORD': 'foo_password', 'OS_TENANT_NAME': 'foo_tenant', 'OS_TENANT_ID': 'foo_tenant_id', 'OS_PROJECT_NAME': 'foo_project', 'OS_PROJECT_ID': 'foo_project_id', 'OS_PROJECT_DOMAIN_ID': 'foo_project_domain_id', 'OS_PROJECT_DOMAIN_NAME': 'foo_project_domain_name', 'OS_PROJECT_DOMAIN_ID': 'foo_project_domain_id', 'OS_USER_DOMAIN_NAME': 'foo_user_domain_name', 'OS_USER_DOMAIN_ID': 'foo_user_domain_id', 'OS_CERT': 'foo_cert', 'OS_KEY': os_key}
    self.set_env_vars(env_vars)
    cert = env_vars['OS_CERT']
    if os_key:
        cert = (cert, env_vars['OS_KEY'])
    with mock.patch.object(shell, 'client') as mock_client:
        self.shell('list')
        mock_client.Client.assert_called_with(manilaclient.API_MAX_VERSION, username=env_vars['OS_USERNAME'], password=env_vars['OS_PASSWORD'], project_name=env_vars['OS_PROJECT_NAME'], auth_url=env_vars['OS_AUTH_URL'], insecure=False, region_name='', tenant_id=env_vars['OS_PROJECT_ID'], endpoint_type='publicURL', extensions=mock.ANY, service_type=constants.V2_SERVICE_TYPE, service_name='', retries=0, http_log_debug=False, cacert=None, use_keyring=False, force_new_token=False, user_id=env_vars['OS_USER_ID'], user_domain_id=env_vars['OS_USER_DOMAIN_ID'], user_domain_name=env_vars['OS_USER_DOMAIN_NAME'], project_domain_id=env_vars['OS_PROJECT_DOMAIN_ID'], project_domain_name=env_vars['OS_PROJECT_DOMAIN_NAME'], cert=cert, input_auth_token='', service_catalog_url='')
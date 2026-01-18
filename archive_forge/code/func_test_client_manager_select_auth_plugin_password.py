import copy
from unittest import mock
from keystoneauth1.access import service_catalog
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1.identity import generic as generic_plugin
from keystoneauth1.identity.v3 import k2k
from keystoneauth1 import loading
from keystoneauth1 import noauth
from keystoneauth1 import token_endpoint
from openstack.config import cloud_config
from openstack.config import defaults
from openstack import connection
from osc_lib.api import auth
from osc_lib import clientmanager
from osc_lib import exceptions as exc
from osc_lib.tests import fakes
from osc_lib.tests import utils
def test_client_manager_select_auth_plugin_password(self):
    auth_args = {'auth_url': fakes.AUTH_URL, 'username': fakes.USERNAME, 'password': fakes.PASSWORD, 'tenant_name': fakes.PROJECT_NAME}
    self._make_clientmanager(auth_args=auth_args, identity_api_version='2.0', auth_plugin_name='v2password')
    auth_args = copy.deepcopy(self.default_password_auth)
    auth_args.update({'user_domain_name': 'default', 'project_domain_name': 'default'})
    self._make_clientmanager(auth_args=auth_args, identity_api_version='3', auth_plugin_name='v3password')
    auth_args = {'auth_url': fakes.AUTH_URL, 'username': fakes.USERNAME, 'password': fakes.PASSWORD, 'tenant_name': fakes.PROJECT_NAME}
    self._make_clientmanager(auth_args=auth_args, identity_api_version='2.0')
    auth_args = copy.deepcopy(self.default_password_auth)
    auth_args.update({'user_domain_name': 'default', 'project_domain_name': 'default'})
    self._make_clientmanager(auth_args=auth_args, identity_api_version='3')
    auth_args = copy.deepcopy(self.default_password_auth)
    auth_args.pop('username')
    auth_args.update({'user_id': fakes.USER_ID})
    self._make_clientmanager(auth_args=auth_args, identity_api_version='3')
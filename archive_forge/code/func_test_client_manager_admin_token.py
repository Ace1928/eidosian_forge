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
def test_client_manager_admin_token(self):
    token_auth = {'endpoint': fakes.AUTH_URL, 'token': fakes.AUTH_TOKEN}
    client_manager = self._make_clientmanager(auth_args=token_auth, auth_plugin_name='admin_token')
    self.assertEqual(fakes.AUTH_URL, client_manager._cli_options.config['auth']['endpoint'])
    self.assertEqual(fakes.AUTH_TOKEN, client_manager.auth.get_token(None))
    self.assertIsInstance(client_manager.auth, token_endpoint.Token)
    self.assertNotEqual(False, client_manager.is_service_available('network'))
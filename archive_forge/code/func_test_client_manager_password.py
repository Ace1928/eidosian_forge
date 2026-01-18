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
def test_client_manager_password(self):
    client_manager = self._make_clientmanager(auth_required=True)
    self.assertEqual(fakes.AUTH_URL, client_manager._cli_options.config['auth']['auth_url'])
    self.assertEqual(fakes.USERNAME, client_manager._cli_options.config['auth']['username'])
    self.assertEqual(fakes.PASSWORD, client_manager._cli_options.config['auth']['password'])
    self.assertIsInstance(client_manager.auth, generic_plugin.Password)
    self.assertTrue(client_manager.verify)
    self.assertIsNone(client_manager.cert)
    self.assertEqual(AUTH_REF.pop('version'), client_manager.auth_ref.version)
    self.assertEqual(fakes.to_unicode_dict(AUTH_REF), client_manager.auth_ref._data['access'])
    self.assertEqual(dir(SERVICE_CATALOG), dir(client_manager.auth_ref.service_catalog))
    self.assertTrue(client_manager.is_service_available('network'))
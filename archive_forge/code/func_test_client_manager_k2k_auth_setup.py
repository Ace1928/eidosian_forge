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
def test_client_manager_k2k_auth_setup(self):
    loader = loading.get_plugin_loader('password')
    auth_plugin = loader.load_from_options(**AUTH_DICT)
    cli_options = defaults.get_defaults()
    cli_options.update({'auth_type': 'password', 'auth': AUTH_DICT, 'interface': fakes.INTERFACE, 'region_name': fakes.REGION_NAME, 'service_provider': fakes.SERVICE_PROVIDER_ID, 'remote_project_id': fakes.PROJECT_ID})
    client_manager = self._clientmanager_class()(cli_options=cloud_config.CloudConfig(name='t1', region='1', config=cli_options, auth_plugin=auth_plugin), api_version={'identity': '3'})
    self.assertFalse(client_manager._auth_setup_completed)
    client_manager.setup_auth()
    self.assertIsInstance(client_manager.auth, k2k.Keystone2Keystone)
    self.assertEqual(client_manager.auth._sp_id, fakes.SERVICE_PROVIDER_ID)
    self.assertEqual(client_manager.auth.project_id, fakes.PROJECT_ID)
    self.assertTrue(client_manager._auth_setup_completed)
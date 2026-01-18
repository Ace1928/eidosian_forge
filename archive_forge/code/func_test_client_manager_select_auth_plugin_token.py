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
def test_client_manager_select_auth_plugin_token(self):
    self._make_clientmanager(identity_api_version='2.0', auth_plugin_name='v2token')
    self._make_clientmanager(identity_api_version='3', auth_plugin_name='v3token')
    self._make_clientmanager(identity_api_version='x', auth_plugin_name='token')
import json
from unittest import mock
import uuid
from keystoneauth1 import access as ks_access
from keystoneauth1 import exceptions as kc_exception
from keystoneauth1.identity import access as ks_auth_access
from keystoneauth1.identity import generic as ks_auth
from keystoneauth1 import loading as ks_loading
from keystoneauth1 import session as ks_session
from keystoneauth1 import token_endpoint as ks_token_endpoint
from keystoneclient.v3 import client as kc_v3
from keystoneclient.v3 import domains as kc_v3_domains
from oslo_config import cfg
from heat.common import config
from heat.common import exception
from heat.common import password_gen
from heat.engine.clients.os.keystone import heat_keystoneclient
from heat.tests import common
from heat.tests import utils
def test_init_v3_token(self):
    """Test creating the client, token auth."""
    self._stubs_auth()
    ctx = utils.dummy_context()
    ctx.username = None
    ctx.password = None
    ctx.trust_id = None
    heat_ks_client = heat_keystoneclient.KeystoneClient(ctx)
    heat_ks_client.client
    self.assertIsNotNone(heat_ks_client._client)
    self._validate_stub_auth()
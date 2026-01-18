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
def test_delete_ec2_keypair_id(self):
    """Test deleting ec2 credential by id."""
    user_id = 'atestuser'
    self._stubs_auth(user_id=user_id)
    ctx = utils.dummy_context()
    ctx.trust_id = None
    credential_id = 'acredential123'
    exc = kc_exception.NotFound
    self.mock_ks_v3_client.credentials.delete.side_effect = exc
    heat_ks_client = heat_keystoneclient.KeystoneClient(ctx)
    self.assertIsNone(heat_ks_client.delete_ec2_keypair(credential_id=credential_id))
    self.assertIsNone(heat_ks_client.delete_ec2_keypair(credential_id=credential_id))
    self.mock_ks_v3_client.credentials.delete.assert_called_with(credential_id)
    self._validate_stub_auth()
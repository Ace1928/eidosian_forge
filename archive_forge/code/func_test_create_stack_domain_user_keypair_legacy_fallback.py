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
def test_create_stack_domain_user_keypair_legacy_fallback(self):
    """Test creating ec2 credentials for domain user, fallback path."""
    self._clear_domain_override()
    self._stubs_auth()
    ctx = utils.dummy_context()
    ctx.trust_id = None
    ex_data = {'access': 'dummy_access2', 'secret': 'dummy_secret2'}
    ex_data_json = json.dumps(ex_data)
    self._stub_gen_creds('dummy_access2', 'dummy_secret2')
    mock_cred = mock.Mock()
    mock_cred.id = '1234567'
    mock_cred.user_id = 'atestuser2'
    mock_cred.blob = ex_data_json
    mock_cred.type = 'ec2'
    self.mock_ks_v3_client.credentials.create.return_value = mock_cred
    heat_ks_client = heat_keystoneclient.KeystoneClient(ctx)
    ec2_cred = heat_ks_client.create_stack_domain_user_keypair(user_id='atestuser2', project_id='aproject')
    self.assertEqual('1234567', ec2_cred.id)
    self.assertEqual('dummy_access2', ec2_cred.access)
    self.assertEqual('dummy_secret2', ec2_cred.secret)
    self.mock_ks_v3_client.credentials.create.assert_called_once_with(user='atestuser2', type='ec2', blob=ex_data_json, project=ctx.tenant_id)
    self._validate_stub_auth()
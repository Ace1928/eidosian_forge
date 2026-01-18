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
def test_create_stack_domain_project_legacy_fallback(self):
    """Test the create_stack_domain_project function, fallback path."""
    self._clear_domain_override()
    ctx = utils.dummy_context()
    ctx.trust_id = None
    self.patchobject(ctx, '_create_auth_plugin')
    heat_ks_client = heat_keystoneclient.KeystoneClient(ctx)
    self.assertEqual(ctx.tenant_id, heat_ks_client.create_stack_domain_project('astack'))
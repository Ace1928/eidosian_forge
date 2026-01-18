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
def test_init_domain_cfg_not_set_error(self):
    """Test error path when config lacks domain config."""
    cfg.CONF.clear_override('stack_domain_admin')
    cfg.CONF.clear_override('stack_domain_admin_password')
    err = self.assertRaises(exception.Error, config.startup_sanity_check)
    exp_msg = 'heat.conf misconfigured, cannot specify "stack_user_domain_id" or "stack_user_domain_name" without "stack_domain_admin" and "stack_domain_admin_password"'
    self.assertIn(exp_msg, str(err))
import os
from unittest import mock
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from oslo_config import fixture as config_fixture
from oslo_middleware import request_id
from oslo_policy import opts as policy_opts
from oslo_utils import importutils
import webob
from heat.common import context
from heat.common import exception
from heat.tests import common
def test_keystone_v3_endpoint_not_set_in_config(self):
    """Ensure an exception is raised when the auth_uri cannot be obtained.

        Ensure an exception is raised when the auth_uri cannot be obtained
        from any source.
        """
    policy_check = 'heat.common.policy.Enforcer.check_is_admin'
    with mock.patch(policy_check) as pc:
        pc.return_value = False
        ctx = context.RequestContext(auth_url=None)
        self.assertRaises(exception.AuthorizationFailure, getattr, ctx, 'keystone_v3_endpoint')
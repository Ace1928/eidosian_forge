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
def test_admin_context_policy_false(self):
    policy_check = 'heat.common.policy.Enforcer.check_is_admin'
    with mock.patch(policy_check) as pc:
        pc.return_value = False
        ctx = context.RequestContext(roles=['notadmin'])
        self.assertFalse(ctx.is_admin)
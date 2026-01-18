import os
from unittest import mock
import yaml
import fixtures
from oslo_config import cfg
from oslo_context import context
from oslo_serialization import jsonutils
from oslotest import base as test_base
from oslo_policy import _cache_handler
from oslo_policy import _checks
from oslo_policy import _parser
from oslo_policy import policy
from oslo_policy.tests import base
def test_enforcer_raises_invalid_scope_with_project_scope_type(self):
    self.conf.set_override('enforce_scope', True, group='oslo_policy')
    rule = policy.RuleDefault(name='fake_rule', check_str='role:test', scope_types=['project'])
    self.enforcer.register_default(rule)
    ctx = context.RequestContext(system_scope='all')
    target_dict = {}
    self.assertRaises(policy.InvalidScope, self.enforcer.enforce, 'fake_rule', target_dict, ctx, True)
    self.assertFalse(self.enforcer.enforce('fake_rule', target_dict, ctx, do_raise=False))
    ctx = context.RequestContext(domain_id='fake')
    self.assertRaises(policy.InvalidScope, self.enforcer.enforce, 'fake_rule', target_dict, ctx, True)
    self.assertFalse(self.enforcer.enforce('fake_rule', target_dict, ctx, do_raise=False))
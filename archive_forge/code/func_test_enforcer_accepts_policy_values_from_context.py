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
def test_enforcer_accepts_policy_values_from_context(self):
    rule = policy.RuleDefault(name='fake_rule', check_str='role:test')
    self.enforcer.register_default(rule)
    request_context = context.RequestContext()
    policy_values = request_context.to_policy_values()
    target_dict = {}
    self.enforcer.enforce('fake_rule', target_dict, policy_values)
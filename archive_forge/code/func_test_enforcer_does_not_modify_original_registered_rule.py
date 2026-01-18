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
def test_enforcer_does_not_modify_original_registered_rule(self):
    rule_original = policy.RuleDefault(name='test', check_str='role:owner')
    self.enforcer.register_default(rule_original)
    self.enforcer.registered_rules['test']._check_str = 'role:admin'
    self.enforcer.registered_rules['test']._check = 'role:admin'
    self.assertEqual(self.enforcer.registered_rules['test'].check_str, 'role:admin')
    self.assertEqual(self.enforcer.registered_rules['test'].check, 'role:admin')
    self.assertEqual(rule_original.check_str, 'role:owner')
    self.assertEqual(rule_original.check.__str__(), 'role:owner')
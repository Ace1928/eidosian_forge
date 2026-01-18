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
def test_enforcer_default_rule_name(self):
    enforcer = policy.Enforcer(self.conf, default_rule='foo_rule')
    self.assertEqual('foo_rule', enforcer.rules.default_rule)
    self.conf.set_override('policy_default_rule', 'bar_rule', group='oslo_policy')
    enforcer = policy.Enforcer(self.conf, default_rule='foo_rule')
    self.assertEqual('foo_rule', enforcer.rules.default_rule)
    enforcer = policy.Enforcer(self.conf)
    self.assertEqual('bar_rule', enforcer.rules.default_rule)
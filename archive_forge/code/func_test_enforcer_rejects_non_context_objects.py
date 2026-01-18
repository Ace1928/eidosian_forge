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
def test_enforcer_rejects_non_context_objects(self):
    rule = policy.RuleDefault(name='fake_rule', check_str='role:test')
    self.enforcer.register_default(rule)

    class InvalidContext(object):
        pass
    request_context = InvalidContext()
    target_dict = {}
    self.assertRaises(policy.InvalidContextObject, self.enforcer.enforce, 'fake_rule', target_dict, request_context)
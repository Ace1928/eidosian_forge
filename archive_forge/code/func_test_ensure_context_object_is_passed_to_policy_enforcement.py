from collections import abc
from unittest import mock
import hashlib
import os.path
import oslo_config.cfg
from oslo_policy import policy as common_policy
import glance.api.policy
from glance.common import exception
import glance.context
from glance.policies import base as base_policy
from glance.tests.unit import base
def test_ensure_context_object_is_passed_to_policy_enforcement(self):
    context = glance.context.RequestContext()
    mock_enforcer = self.mock_object(common_policy.Enforcer, 'enforce')
    enforcer = glance.api.policy.Enforcer(suppress_deprecation_warnings=True)
    enforcer.register_default(common_policy.RuleDefault(name='foo', check_str='role:bar'))
    enforcer.enforce(context, 'foo', {})
    mock_enforcer.assert_called_once_with('foo', {}, context, do_raise=True, exc=exception.Forbidden, action='foo')
    mock_enforcer.reset_mock()
    enforcer.check(context, 'foo', {})
    mock_enforcer.assert_called_once_with('foo', {}, context)
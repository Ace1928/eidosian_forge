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
def test_deprecate_a_policy_name(self):
    deprecated_rule = policy.DeprecatedRule(name='foo:bar', check_str='role:baz', deprecated_reason='"foo:bar" is not granular enough. If your deployment has overridden "foo:bar", ensure you override the new policies with same role or rule. Not doing this will require the service to assume the new defaults for "foo:bar:create", "foo:bar:update", "foo:bar:list", and "foo:bar:delete", which might be backwards incompatible for your deployment', deprecated_since='N')
    rule_list = [policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:baz', description='Create a bar.', operations=[{'path': '/v1/bars/', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
    expected_msg = 'Policy "foo:bar":"role:baz" was deprecated in N in favor of "foo:create_bar":"role:baz". Reason: "foo:bar" is not granular enough. If your deployment has overridden "foo:bar", ensure you override the new policies with same role or rule. Not doing this will require the service to assume the new defaults for "foo:bar:create", "foo:bar:update", "foo:bar:list", and "foo:bar:delete", which might be backwards incompatible for your deployment. Either ensure your deployment is ready for the new default or copy/paste the deprecated policy into your policy file and maintain it manually.'
    rules = jsonutils.dumps({'foo:bar': 'role:bang'})
    self.create_config_file('policy.json', rules)
    enforcer = policy.Enforcer(self.conf)
    enforcer.register_defaults(rule_list)
    with mock.patch('warnings.warn') as mock_warn:
        enforcer.load_rules(True)
        mock_warn.assert_called_once_with(expected_msg)
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
def test_deprecate_name_suppress_does_not_log_warning(self):
    deprecated_rule = policy.DeprecatedRule(name='foo:bar', check_str='role:baz', deprecated_reason='"foo:bar" is not granular enough.', deprecated_since='N')
    rule_list = [policy.DocumentedRuleDefault(name='foo:create_bar', check_str='role:baz', description='Create a bar.', operations=[{'path': '/v1/bars/', 'method': 'POST'}], deprecated_rule=deprecated_rule)]
    rules = jsonutils.dumps({'foo:bar': 'role:bang'})
    self.create_config_file('policy.json', rules)
    enforcer = policy.Enforcer(self.conf)
    enforcer.suppress_deprecation_warnings = True
    enforcer.register_defaults(rule_list)
    with mock.patch('warnings.warn') as mock_warn:
        enforcer.load_rules()
        mock_warn.assert_not_called()
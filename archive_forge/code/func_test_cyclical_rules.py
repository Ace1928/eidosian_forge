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
@mock.patch.object(policy, 'LOG')
def test_cyclical_rules(self, mock_log):
    rules = jsonutils.dumps({'foo': 'rule:bar', 'bar': 'rule:foo'})
    self.create_config_file('policy.json', rules)
    self.enforcer.load_rules(True)
    self.assertFalse(self.enforcer.check_rules())
    mock_log.warning.assert_called()
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
def test_load_json_file_log_warning(self, mock_log):
    rules = jsonutils.dumps({'foo': 'rule:bar'})
    self.create_config_file('policy.json', rules)
    self.enforcer.load_rules(True)
    mock_log.warning.assert_any_call(policy.WARN_JSON)
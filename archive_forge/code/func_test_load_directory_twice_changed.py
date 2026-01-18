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
@mock.patch('oslo_policy.policy.Enforcer.check_rules')
def test_load_directory_twice_changed(self, mock_check_rules):
    self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
    self.enforcer.load_rules()
    conf_path = os.path.join(self.config_dir, os.path.join('policy.d', 'a.conf'))
    stinfo = os.stat(conf_path)
    os.utime(conf_path, (stinfo.st_atime + 10, stinfo.st_mtime + 10))
    self.enforcer.load_rules()
    self.assertEqual(2, mock_check_rules.call_count)
    self.assertIsNotNone(self.enforcer.rules)
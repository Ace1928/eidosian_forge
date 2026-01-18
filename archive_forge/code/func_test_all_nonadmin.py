import copy
from unittest import mock
from oslo_serialization import jsonutils
from oslo_policy import shell
from oslo_policy.tests import base
from oslo_policy.tests import token_fixture
def test_all_nonadmin(self):
    policy_file = self.get_config_file_fullname('policy.yaml')
    access_file = self.get_config_file_fullname('access.json')
    apply_rule = None
    is_admin = False
    stdout = self._capture_stdout()
    shell.tool(policy_file, access_file, apply_rule, is_admin)
    expected = 'passed: sampleservice:sample_rule\n'
    self.assertEqual(expected, stdout.getvalue())
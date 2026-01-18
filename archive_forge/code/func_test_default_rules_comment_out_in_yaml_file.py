import operator
from unittest import mock
import warnings
from oslo_config import cfg
import stevedore
import testtools
import yaml
from oslo_policy import generator
from oslo_policy import policy
from oslo_policy.tests import base
from oslo_serialization import jsonutils
def test_default_rules_comment_out_in_yaml_file(self):
    converted_policy_data = self._test_convert_json_to_yaml_file()
    commented_default_rule = '# test_rule1\n# GET  /test\n# Intended scope(s): system\n#"rule1_name": "rule:admin"\n\n'
    self.assertIn(commented_default_rule, converted_policy_data)
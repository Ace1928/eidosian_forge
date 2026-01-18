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
def test_existing_deprecated_rules_kept_uncommented_in_yaml_file(self):
    converted_policy_data = self._test_convert_json_to_yaml_file()
    existing_deprecated_rule_with_warning = '# WARNING: Below rules are either deprecated rules\n# or extra rules in policy file, it is strongly\n# recommended to switch to new rules.\n"deprecated_rule1_name": "rule:admin"\n'
    self.assertIn(existing_deprecated_rule_with_warning, converted_policy_data)
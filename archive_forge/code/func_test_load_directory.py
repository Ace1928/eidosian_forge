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
def test_load_directory(self):
    self.create_config_file('policy.d/a.conf', POLICY_JSON_CONTENTS)
    self.create_config_file('policy.d/b.conf', POLICY_B_CONTENTS)
    self.enforcer.load_rules(True)
    self.assertIsNotNone(self.enforcer.rules)
    loaded_rules = jsonutils.loads(str(self.enforcer.rules))
    self.assertEqual('role:fakeB', loaded_rules['default'])
    self.assertEqual('is_admin:True', loaded_rules['admin'])
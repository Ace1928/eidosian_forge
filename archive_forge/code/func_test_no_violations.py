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
def test_no_violations(self):
    self.create_config_file('policy.json', POLICY_JSON_CONTENTS)
    self.enforcer.load_rules(True)
    self.assertTrue(self.enforcer.check_rules(raise_on_violation=True))
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
def test_non_reversible_check(self):
    self.create_config_file('policy.json', jsonutils.dumps({'shared': 'field:networks:shared=True'}))
    self.enforcer.load_rules(True)
    self.assertIsNotNone(self.enforcer.rules)
    loaded_rules = jsonutils.loads(str(self.enforcer.rules))
    self.assertNotEqual('field:networks:shared=True', loaded_rules['shared'])
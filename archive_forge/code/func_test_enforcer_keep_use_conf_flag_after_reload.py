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
def test_enforcer_keep_use_conf_flag_after_reload(self):
    self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
    self.create_config_file(os.path.join('policy.d', 'b.conf'), POLICY_B_CONTENTS)
    self.assertTrue(self.enforcer.use_conf)
    self.assertTrue(self.enforcer.enforce('default', {}, {'roles': ['fakeB']}))
    self.assertFalse(self.enforcer.enforce('test', {}, {'roles': ['test']}))
    self.assertTrue(self.enforcer.use_conf)
    self.assertFalse(self.enforcer.enforce('_dynamic_test_rule', {}, {'roles': ['test']}))
    rules = jsonutils.loads(str(self.enforcer.rules))
    rules['_dynamic_test_rule'] = 'role:test'
    with open(self.enforcer.policy_path, 'w') as f:
        f.write(jsonutils.dumps(rules))
    self.enforcer.load_rules(force_reload=True)
    self.assertTrue(self.enforcer.enforce('_dynamic_test_rule', {}, {'roles': ['test']}))
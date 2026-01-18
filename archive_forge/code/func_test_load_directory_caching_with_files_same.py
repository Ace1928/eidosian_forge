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
def test_load_directory_caching_with_files_same(self, overwrite=True):
    self.enforcer.overwrite = overwrite
    self.create_config_file(os.path.join('policy.d', 'a.conf'), POLICY_A_CONTENTS)
    self.enforcer.load_rules(False)
    self.assertIsNotNone(self.enforcer.rules)
    old = next(iter(self.enforcer._policy_dir_mtimes))
    self.assertEqual(1, len(self.enforcer._policy_dir_mtimes))
    self.enforcer.load_rules(False)
    self.assertEqual(1, len(self.enforcer._policy_dir_mtimes))
    self.assertEqual(old, next(iter(self.enforcer._policy_dir_mtimes)))
    loaded_rules = jsonutils.loads(str(self.enforcer.rules))
    self.assertEqual('is_admin:True', loaded_rules['admin'])
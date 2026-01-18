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
def test_get_policy_path_raises_exc(self):
    enforcer = policy.Enforcer(self.conf, policy_file='raise_error.json')
    e = self.assertRaises(cfg.ConfigFilesNotFoundError, enforcer._get_policy_path, enforcer.policy_file)
    self.assertEqual(('raise_error.json',), e.config_files)
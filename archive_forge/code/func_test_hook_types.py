import os.path
from unittest import mock
import fixtures
from oslo_config import cfg
from heat.common import environment_format
from heat.common import exception
from heat.engine import environment
from heat.engine import resources
from heat.engine.resources.aws.ec2 import instance
from heat.engine.resources.openstack.nova import server
from heat.engine import support
from heat.tests import common
from heat.tests import generic_resource
from heat.tests import utils
def test_hook_types(self):
    resources = {u'hook': {u'hooks': self.hook}, u'not-hook': {u'hooks': [hook for hook in environment.HOOK_TYPES if hook != self.hook]}, u'all': {u'hooks': environment.HOOK_TYPES}}
    registry = environment.ResourceRegistry(None, {})
    registry.load({'resources': resources})
    self.assertTrue(registry.matches_hook('hook', self.hook))
    self.assertFalse(registry.matches_hook('not-hook', self.hook))
    self.assertTrue(registry.matches_hook('all', self.hook))
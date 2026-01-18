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
def test_load_registry_invalid_hook_type(self):
    resources = {u'resources': {u'a': {u'hooks': 'invalid-type'}}}
    registry = environment.ResourceRegistry(None, {})
    msg = 'Invalid hook type "invalid-type" for resource breakpoint, acceptable hook types are: (\'pre-create\', \'pre-update\', \'pre-delete\', \'post-create\', \'post-update\', \'post-delete\')'
    ex = self.assertRaises(exception.InvalidBreakPointHook, registry.load, {'resources': resources})
    self.assertEqual(msg, str(ex))
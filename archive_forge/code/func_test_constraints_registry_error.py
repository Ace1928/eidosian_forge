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
def test_constraints_registry_error(self):
    constraint_content = '\ndef constraint_mapping():\n    raise ValueError("oops")\n        '
    plugin_dir = self.useFixture(fixtures.TempDir())
    plugin_file = os.path.join(plugin_dir.path, 'test.py')
    with open(plugin_file, 'w+') as ef:
        ef.write(constraint_content)
    cfg.CONF.set_override('plugin_dirs', plugin_dir.path)
    env = environment.Environment({})
    error = self.assertRaises(ValueError, resources._load_global_environment, env)
    self.assertEqual('oops', str(error))
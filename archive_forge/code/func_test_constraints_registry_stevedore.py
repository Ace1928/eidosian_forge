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
def test_constraints_registry_stevedore(self):
    env = environment.Environment({})
    resources._load_global_environment(env)
    self.assertEqual('FlavorConstraint', env.get_constraint('nova.flavor').__name__)
    self.assertIsNone(env.get_constraint('no_constraint'))
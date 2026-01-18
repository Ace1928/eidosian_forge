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
def test_load_old_parameters(self):
    old = {u'a': u'ff', u'b': u'ss'}
    expected = {u'parameters': old, u'encrypted_param_names': [], u'parameter_defaults': {}, u'event_sinks': [], u'resource_registry': {u'resources': {}}}
    env = environment.Environment(old)
    self.assertEqual(expected, env.env_as_dict())
    del expected['encrypted_param_names']
    self.assertEqual(expected, env.user_env_as_dict())
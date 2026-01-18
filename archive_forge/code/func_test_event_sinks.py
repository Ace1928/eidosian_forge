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
def test_event_sinks(self):
    env = environment.Environment({'event_sinks': [{'type': 'zaqar-queue', 'target': 'myqueue'}]})
    self.assertEqual([{'type': 'zaqar-queue', 'target': 'myqueue'}], env.user_env_as_dict()['event_sinks'])
    sinks = env.get_event_sinks()
    self.assertEqual(1, len(sinks))
    self.assertEqual('myqueue', sinks[0]._target)
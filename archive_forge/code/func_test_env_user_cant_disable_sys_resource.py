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
def test_env_user_cant_disable_sys_resource(self):
    u_env_content = '\n        resource_registry:\n            "AWS::*":\n        '
    u_env = environment.Environment()
    u_env.load(environment_format.parse(u_env_content))
    self.assertEqual(instance.Instance, u_env.get_resource_info('AWS::EC2::Instance').value)
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
def test_empty_env_dir(self):
    with mock.patch('glob.glob') as m_ldir:
        m_ldir.return_value = []
        env_dir = '/etc_etc/heat/environment.d'
        env = environment.Environment({}, user_env=False)
        environment.read_global_environment(env, env_dir)
    m_ldir.assert_called_once_with(env_dir + '/*')
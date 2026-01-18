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
def test_happy_path(self):
    with mock.patch('glob.glob') as m_ldir:
        m_ldir.return_value = ['/etc_etc/heat/environment.d/a.yaml']
        env_dir = '/etc_etc/heat/environment.d'
        env_content = '{"resource_registry": {}}'
        env = environment.Environment({}, user_env=False)
        with mock.patch('heat.engine.environment.open', mock.mock_open(read_data=env_content), create=True) as m_open:
            environment.read_global_environment(env, env_dir)
    m_ldir.assert_called_once_with(env_dir + '/*')
    m_open.assert_called_once_with('%s/a.yaml' % env_dir)
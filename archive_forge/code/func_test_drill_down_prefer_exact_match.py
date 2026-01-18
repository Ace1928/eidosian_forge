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
def test_drill_down_prefer_exact_match(self):
    env = {u'resource_registry': {u'resources': {u'*esource': {u'hooks': 'pre-create'}, u'res*': {u'hooks': 'pre-create'}, u'resource': {u'OS::Fruit': u'carrots.yaml', u'hooks': 'pre-update'}, u'resource*': {u'hooks': 'pre-create'}, u'*resource': {u'hooks': 'pre-create'}, u'*sour*': {u'hooks': 'pre-create'}}}}
    penv = environment.Environment(env)
    cenv = environment.get_child_environment(penv, None, child_resource_name=u'resource')
    registry = cenv.user_env_as_dict()['resource_registry']
    resources = registry['resources']
    self.assertEqual(u'carrots.yaml', resources[u'OS::Fruit'])
    self.assertEqual('pre-update', resources[u'hooks'])
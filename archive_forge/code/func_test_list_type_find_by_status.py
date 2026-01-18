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
def test_list_type_find_by_status(self):
    registry = resources.global_env().registry
    types = registry.get_types(support_status=support.UNSUPPORTED)
    self.assertIn('ResourceTypeUnSupportedLiberty', types)
    self.assertNotIn('GenericResourceType', types)
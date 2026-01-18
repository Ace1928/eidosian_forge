import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.aws.lb import loadbalancer as lb
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_child_template_without_key_name(self):
    rsrc = self.setup_loadbalancer(False)
    parsed_template = {'Resources': {'LB_instance': {'Properties': {'KeyName': 'foo'}}}, 'Parameters': {'KeyName': 'foo'}}
    rsrc.get_parsed_template = mock.Mock(return_value=parsed_template)
    tmpl = rsrc.child_template()
    self.assertNotIn('KeyName', tmpl['Parameters'])
    self.assertNotIn('KeyName', tmpl['Resources']['LB_instance']['Properties'])
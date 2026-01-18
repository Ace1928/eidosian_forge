import copy
from unittest import mock
from neutronclient.common import exceptions as q_exceptions
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine import resource
from heat.engine.resources.aws.ec2 import eip
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_eip_with_exception(self):
    self.mock_list_net.return_value = {'networks': [{'status': 'ACTIVE', 'subnets': [], 'name': 'nova', 'router:external': True, 'tenant_id': 'c1210485b2424d48804aad5d39c61b8f', 'admin_state_up': True, 'shared': True, 'id': 'eeee'}]}
    self.patchobject(neutronclient.Client, 'create_floatingip', side_effect=neutronclient.exceptions.NotFound)
    t = template_format.parse(eip_template)
    stack = utils.parse_stack(t)
    resource_name = 'IPAddress'
    resource_defns = stack.t.resource_definitions(stack)
    rsrc = eip.ElasticIp(resource_name, resource_defns[resource_name], stack)
    self.assertRaises(neutronclient.exceptions.NotFound, rsrc.handle_create)
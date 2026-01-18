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
def test_eip_update(self):
    server_old = self.fc.servers.list()[0]
    self.patchobject(self.fc.servers, 'get', return_value=server_old)
    iface = self.mock_interface('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', '1.2.3.4')
    self.patchobject(server_old, 'interface_list', return_value=[iface])
    self.mock_create_floatingip()
    t = template_format.parse(eip_template)
    stack = utils.parse_stack(t)
    rsrc = self.create_eip(t, stack, 'IPAddress')
    self.assertEqual('11.0.0.1', rsrc.FnGetRefId())
    server_update = self.fc.servers.list()[1]
    self.patchobject(self.fc.servers, 'get', return_value=server_update)
    self.patchobject(server_update, 'interface_list', return_value=[iface])
    props = copy.deepcopy(rsrc.properties.data)
    update_server_id = '5678'
    props['InstanceId'] = update_server_id
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    scheduler.TaskRunner(rsrc.update, update_snippet)()
    self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
    self.assertEqual('11.0.0.1', rsrc.FnGetRefId())
    props = copy.deepcopy(rsrc.properties.data)
    props.pop('InstanceId')
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    scheduler.TaskRunner(rsrc.update, update_snippet)()
    self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
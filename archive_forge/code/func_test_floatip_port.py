import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from openstack import exceptions
from oslo_utils import excutils
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.engine.clients.os import neutron
from heat.engine.hot import functions as hot_funcs
from heat.engine import node_data
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import stk_defn
from heat.engine import template as tmpl
from heat.tests import common
from heat.tests import utils
def test_floatip_port(self):
    t = template_format.parse(neutron_floating_no_assoc_template)
    t['resources']['port_floating']['properties']['network'] = 'xyz1234'
    t['resources']['port_floating']['properties']['fixed_ips'][0]['subnet'] = 'sub1234'
    t['resources']['router_interface']['properties']['subnet'] = 'sub1234'
    stack = utils.parse_stack(t)
    self.mockclient.create_port.return_value = {'port': {'status': 'BUILD', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}
    self.mockclient.show_port.side_effect = [{'port': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}, qe.PortNotFoundClient(status_code=404)]
    self.mockclient.create_floatingip.return_value = {'floatingip': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}
    self.mockclient.update_floatingip.return_value = {'floatingip': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}
    self.mockclient.delete_floatingip.return_value = None
    self.mockclient.delete_port.return_value = None
    self.mockclient.show_floatingip.side_effect = qe.PortNotFoundClient(status_code=404)
    self.stub_PortConstraint_validate()
    required_by = set(stack.dependencies.required_by(stack['router_interface']))
    self.assertIn(stack['floating_ip'], required_by)
    p = stack['port_floating']
    scheduler.TaskRunner(p.create)()
    self.assertEqual((p.CREATE, p.COMPLETE), p.state)
    stk_defn.update_resource_data(stack.defn, p.name, p.node_data())
    fip = stack['floating_ip']
    scheduler.TaskRunner(fip.create)()
    self.assertEqual((fip.CREATE, fip.COMPLETE), fip.state)
    stk_defn.update_resource_data(stack.defn, fip.name, fip.node_data())
    props = copy.deepcopy(fip.properties.data)
    update_port_id = '2146dfbf-ba77-4083-8e86-d052f671ece5'
    props['port_id'] = update_port_id
    update_snippet = rsrc_defn.ResourceDefinition(fip.name, fip.type(), stack.t.parse(stack.defn, props))
    scheduler.TaskRunner(fip.update, update_snippet)()
    self.assertEqual((fip.UPDATE, fip.COMPLETE), fip.state)
    stk_defn.update_resource_data(stack.defn, fip.name, fip.node_data())
    props = copy.deepcopy(fip.properties.data)
    del props['port_id']
    update_snippet = rsrc_defn.ResourceDefinition(fip.name, fip.type(), stack.t.parse(stack.defn, props))
    scheduler.TaskRunner(fip.update, update_snippet)()
    self.assertEqual((fip.UPDATE, fip.COMPLETE), fip.state)
    scheduler.TaskRunner(fip.delete)()
    scheduler.TaskRunner(p.delete)()
    self.mockclient.create_port.assert_called_once_with({'port': {'network_id': u'xyz1234', 'fixed_ips': [{'subnet_id': u'sub1234', 'ip_address': u'10.0.0.10'}], 'name': utils.PhysName(stack.name, 'port_floating'), 'admin_state_up': True, 'binding:vnic_type': 'normal', 'device_owner': '', 'device_id': ''}})
    self.mockclient.show_port.assert_called_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766')
    self.mockclient.create_floatingip.assert_called_once_with({'floatingip': {'floating_network_id': u'abcd1234', 'port_id': u'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}})
    self.mockclient.update_floatingip.assert_has_calls([mock.call('fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'floatingip': {'port_id': u'2146dfbf-ba77-4083-8e86-d052f671ece5', 'fixed_ip_address': None}}), mock.call('fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'floatingip': {'port_id': None, 'fixed_ip_address': None}})])
    self.mockclient.delete_floatingip.assert_called_once_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766')
    self.mockclient.show_floatingip.assert_called_once_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766')
    self.mockclient.delete_port.assert_called_once_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766')
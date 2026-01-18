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
def test_floatip_association_port(self):
    t = template_format.parse(neutron_floating_template)
    stack = utils.parse_stack(t)
    self.mockclient.create_floatingip.return_value = {'floatingip': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}
    self.mockclient.create_port.return_value = {'port': {'status': 'BUILD', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}
    self.mockclient.show_port.side_effect = [{'port': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}, qe.PortNotFoundClient(status_code=404)]
    self.mockclient.update_floatingip.side_effect = [{'floatingip': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}, {'floatingip': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}, None, {'floatingip': {'status': 'ACTIVE', 'id': '2146dfbf-ba77-4083-8e86-d052f671ece5'}}, None, {'floatingip': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}, None]
    self.mockclient.delete_port.side_effect = [None, qe.PortNotFoundClient(status_code=404)]
    self.mockclient.delete_floatingip.side_effect = [None, qe.PortNotFoundClient(status_code=404)]
    self.mockclient.show_floatingip.side_effect = qe.NeutronClientException(status_code=404)
    self.stub_PortConstraint_validate()
    fip = stack['floating_ip']
    scheduler.TaskRunner(fip.create)()
    self.assertEqual((fip.CREATE, fip.COMPLETE), fip.state)
    stk_defn.update_resource_data(stack.defn, fip.name, fip.node_data())
    p = stack['port_floating']
    scheduler.TaskRunner(p.create)()
    self.assertEqual((p.CREATE, p.COMPLETE), p.state)
    stk_defn.update_resource_data(stack.defn, p.name, p.node_data())
    fipa = stack['floating_ip_assoc']
    scheduler.TaskRunner(fipa.create)()
    self.assertEqual((fipa.CREATE, fipa.COMPLETE), fipa.state)
    stk_defn.update_resource_data(stack.defn, fipa.name, fipa.node_data())
    self.assertIsNotNone(fipa.id)
    self.assertEqual(fipa.id, fipa.resource_id)
    fipa.validate()
    props = copy.deepcopy(fipa.properties.data)
    update_port_id = '2146dfbf-ba77-4083-8e86-d052f671ece5'
    props['port_id'] = update_port_id
    update_snippet = rsrc_defn.ResourceDefinition(fipa.name, fipa.type(), stack.t.parse(stack.defn, props))
    scheduler.TaskRunner(fipa.update, update_snippet)()
    self.assertEqual((fipa.UPDATE, fipa.COMPLETE), fipa.state)
    props = copy.deepcopy(fipa.properties.data)
    update_flip_id = '2146dfbf-ba77-4083-8e86-d052f671ece5'
    props['floatingip_id'] = update_flip_id
    update_snippet = rsrc_defn.ResourceDefinition(fipa.name, fipa.type(), props)
    scheduler.TaskRunner(fipa.update, update_snippet)()
    self.assertEqual((fipa.UPDATE, fipa.COMPLETE), fipa.state)
    props = copy.deepcopy(fipa.properties.data)
    update_flip_id = 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'
    update_port_id = 'ade6fcac-7d47-416e-a3d7-ad12efe445c1'
    props['floatingip_id'] = update_flip_id
    props['port_id'] = update_port_id
    update_snippet = rsrc_defn.ResourceDefinition(fipa.name, fipa.type(), props)
    scheduler.TaskRunner(fipa.update, update_snippet)()
    self.assertEqual((fipa.UPDATE, fipa.COMPLETE), fipa.state)
    scheduler.TaskRunner(fipa.delete)()
    scheduler.TaskRunner(p.delete)()
    scheduler.TaskRunner(fip.delete)()
    fip.state_set(fip.CREATE, fip.COMPLETE, 'to delete again')
    p.state_set(p.CREATE, p.COMPLETE, 'to delete again')
    self.assertIsNone(scheduler.TaskRunner(p.delete)())
    scheduler.TaskRunner(fip.delete)()
    self.mockclient.create_floatingip.assert_called_once_with({'floatingip': {'floating_network_id': u'abcd1234'}})
    self.mockclient.create_port.assert_called_once_with({'port': {'network_id': u'abcd1234', 'fixed_ips': [{'subnet_id': u'sub1234', 'ip_address': u'10.0.0.10'}], 'name': utils.PhysName(stack.name, 'port_floating'), 'admin_state_up': True, 'device_owner': '', 'device_id': '', 'binding:vnic_type': 'normal'}})
    self.mockclient.show_port.assert_called_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766')
    self.mockclient.update_floatingip.assert_has_calls([mock.call('fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'floatingip': {'port_id': u'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}), mock.call('fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'floatingip': {'port_id': u'2146dfbf-ba77-4083-8e86-d052f671ece5', 'fixed_ip_address': None}}), mock.call('fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'floatingip': {'port_id': None}}), mock.call('2146dfbf-ba77-4083-8e86-d052f671ece5', {'floatingip': {'port_id': u'2146dfbf-ba77-4083-8e86-d052f671ece5', 'fixed_ip_address': None}}), mock.call('2146dfbf-ba77-4083-8e86-d052f671ece5', {'floatingip': {'port_id': None}}), mock.call('fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'floatingip': {'port_id': u'ade6fcac-7d47-416e-a3d7-ad12efe445c1', 'fixed_ip_address': None}}), mock.call('fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'floatingip': {'port_id': None}})])
    self.mockclient.delete_port.assert_called_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766')
    self.mockclient.delete_floatingip.assert_called_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766')
    self.mockclient.show_floatingip.assert_called_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766')
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from oslo_serialization import jsonutils
from heat.common import exception
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_missing_subnet_id(self):
    t = template_format.parse(neutron_port_template)
    t['resources']['port']['properties']['fixed_ips'][0].pop('subnet')
    stack = utils.parse_stack(t)
    self.find_mock.return_value = 'net1234'
    self.create_mock.return_value = {'port': {'status': 'BUILD', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}
    self.port_show_mock.return_value = {'port': {'status': 'ACTIVE', 'id': 'fc68ea2c-b60b-4b4f-bd82-94ec81110766'}}
    port = stack['port']
    scheduler.TaskRunner(port.create)()
    self.assertEqual((port.CREATE, port.COMPLETE), port.state)
    self.create_mock.assert_called_once_with({'port': {'network_id': u'net1234', 'fixed_ips': [{'ip_address': u'10.0.3.21'}], 'name': utils.PhysName(stack.name, 'port'), 'admin_state_up': True, 'device_owner': u'network:dhcp', 'binding:vnic_type': 'normal', 'device_id': ''}})
    self.port_show_mock.assert_called_once_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766')
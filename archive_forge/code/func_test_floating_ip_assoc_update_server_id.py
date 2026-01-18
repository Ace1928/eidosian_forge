import copy
from unittest import mock
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception as heat_ex
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.openstack.nova import floatingip
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_floating_ip_assoc_update_server_id(self):
    rsrc = self.prepare_floating_ip_assoc()
    rsrc.validate()
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    return_server = self.novaclient.servers.list()[2]
    self.patchobject(self.novaclient.servers, 'get', return_value=return_server)
    iface = self.mock_interface('bbbbb-bbbb-bbbb-bbbbbbbbb', '4.5.6.7')
    self.patchobject(return_server, 'interface_list', return_value=[iface])
    props = copy.deepcopy(rsrc.properties.data)
    update_server_id = '2146dfbf-ba77-4083-8e86-d052f671ece5'
    props['server_id'] = update_server_id
    update_snippet = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    scheduler.TaskRunner(rsrc.update, update_snippet)()
    self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
    fip_request = {'floatingip': {'fixed_ip_address': '1.2.3.4', 'port_id': 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'}}
    fip_request_update = {'floatingip': {'fixed_ip_address': '4.5.6.7', 'port_id': 'bbbbb-bbbb-bbbb-bbbbbbbbb'}}
    calls = [mock.call('fc68ea2c-b60b-4b4f-bd82-94ec81110766', fip_request), mock.call('fc68ea2c-b60b-4b4f-bd82-94ec81110766', fip_request_update)]
    self.mock_upd_fip.assert_has_calls(calls)
    self.assertEqual(2, self.mock_upd_fip.call_count)
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.common import timeutils
from heat.engine.resources.openstack.neutron import network_gateway
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_network_gateway_update(self):
    rsrc = self.prepare_create_network_gateway()
    self.mockclient.update_network_gateway.return_value = None
    self.mockclient.disconnect_network_gateway.side_effect = [None, qe.NeutronClientException(status_code=404), None]
    self.mockclient.connect_network_gateway.side_effect = [self.mockclient.connect_network_gateway.return_value, {'connection_info': {'network_gateway_id': 'ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', 'network_id': '6af055d3-26f6-48dd-a597-7611d7e58d35', 'port_id': 'aa800972-f6be-4c65-8453-9ab31834bf80'}}, {'connection_info': {'network_gateway_id': 'ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', 'network_id': '6af055d3-26f6-48dd-a597-7611d7e58d35', 'port_id': 'aa800972-f6be-4c65-8453-9ab31834bf80'}}, {'connection_info': {'network_gateway_id': 'ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', 'network_id': '6af055d3-26f6-48dd-a597-7611d7e58d35', 'port_id': 'aa800972-f6be-4c65-8453-9ab31834bf80'}}, {'connection_info': {'network_gateway_id': 'ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', 'network_id': '6af055d3-26f6-48dd-a597-7611d7e58d35', 'port_id': 'aa800972-f6be-4c65-8453-9ab31834bf80'}}]
    self.mockclient.delete_network_gateway.return_value = None
    self.mockclient.create_network_gateway.side_effect = [self.mockclient.create_network_gateway.return_value, {'network_gateway': {'id': 'ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', 'name': 'NetworkGateway', 'default': False, 'tenant_id': '96ba52dc-c5c5-44c6-9a9d-d3ba1a03f77f', 'devices': [{'id': 'e52148ca-7db9-4ec3-abe6-2c7c0ff316eb', 'interface_name': 'breth2'}]}}]
    rsrc.validate()
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    snippet_for_update1 = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), {'name': u'NetworkGatewayUpdate', 'devices': [{'id': u'e52148ca-7db9-4ec3-abe6-2c7c0ff316eb', 'interface_name': u'breth1'}], 'connections': [{'network': '6af055d3-26f6-48dd-a597-7611d7e58d35', 'segmentation_type': 'vlan', 'segmentation_id': 10}]})
    scheduler.TaskRunner(rsrc.update, snippet_for_update1)()
    snippet_for_update2 = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), {'name': u'NetworkGatewayUpdate', 'devices': [{'id': u'e52148ca-7db9-4ec3-abe6-2c7c0ff316eb', 'interface_name': u'breth1'}], 'connections': [{'network': u'6af055d3-26f6-48dd-a597-7611d7e58d35', 'segmentation_type': u'flat', 'segmentation_id': 0}]})
    scheduler.TaskRunner(rsrc.update, snippet_for_update2, snippet_for_update1)()
    snippet_for_update3 = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), {'name': u'NetworkGatewayUpdate', 'devices': [{'id': u'e52148ca-7db9-4ec3-abe6-2c7c0ff316eb', 'interface_name': u'breth1'}], 'connections': [{'network': u'6af055d3-26f6-48dd-a597-7611d7e58d35', 'segmentation_type': u'flat', 'segmentation_id': 1}]})
    scheduler.TaskRunner(rsrc.update, snippet_for_update3, snippet_for_update2)()
    snippet_for_update4 = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), {'name': u'NetworkGatewayUpdate', 'devices': [{'id': u'e52148ca-7db9-4ec3-abe6-2c7c0ff316eb', 'interface_name': u'breth2'}], 'connections': [{'network_id': u'6af055d3-26f6-48dd-a597-7611d7e58d35', 'segmentation_type': u'vlan', 'segmentation_id': 10}]})
    scheduler.TaskRunner(rsrc.update, snippet_for_update4, snippet_for_update3)()
    self.mockclient.create_network_gateway.assert_has_calls([mock.call({'network_gateway': {'name': 'NetworkGateway', 'devices': [{'id': 'e52148ca-7db9-4ec3-abe6-2c7c0ff316eb', 'interface_name': 'breth1'}]}}), mock.call({'network_gateway': {'name': u'NetworkGatewayUpdate', 'devices': [{'id': u'e52148ca-7db9-4ec3-abe6-2c7c0ff316eb', 'interface_name': u'breth2'}]}})])
    self.mockclient.connect_network_gateway.assert_has_calls([mock.call('ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', {'network_id': '6af055d3-26f6-48dd-a597-7611d7e58d35', 'segmentation_id': 10, 'segmentation_type': 'vlan'}), mock.call('ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', {'network_id': '6af055d3-26f6-48dd-a597-7611d7e58d35', 'segmentation_id': 0, 'segmentation_type': 'flat'}), mock.call('ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', {'network_id': '6af055d3-26f6-48dd-a597-7611d7e58d35', 'segmentation_id': 1, 'segmentation_type': 'flat'}), mock.call('ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', {'network_id': u'6af055d3-26f6-48dd-a597-7611d7e58d35', 'segmentation_id': 1, 'segmentation_type': u'flat'})])
    self.mockclient.update_network_gateway.assert_has_calls([mock.call('ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', {'network_gateway': {'name': 'NetworkGatewayUpdate'}})])
    self.mockclient.disconnect_network_gateway.assert_has_calls([mock.call('ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', {'network_id': '6af055d3-26f6-48dd-a597-7611d7e58d35', 'segmentation_id': 10, 'segmentation_type': 'vlan'}), mock.call('ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', {'network_id': '6af055d3-26f6-48dd-a597-7611d7e58d35', 'segmentation_id': 0, 'segmentation_type': 'flat'}), mock.call('ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', {'network_id': '6af055d3-26f6-48dd-a597-7611d7e58d35', 'segmentation_id': 1, 'segmentation_type': 'flat'})])
    self.mockclient.delete_network_gateway.assert_called_once_with('ed4c03b9-8251-4c09-acc4-e59ee9e6aa37')
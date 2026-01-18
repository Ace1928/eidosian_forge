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
def test_network_gateway_attribute(self):
    rsrc = self.prepare_create_network_gateway()
    self.mockclient.show_network_gateway.return_value = sng
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual(u'ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', rsrc.FnGetRefId())
    self.assertFalse(rsrc.FnGetAtt('default'))
    error = self.assertRaises(exception.InvalidTemplateAttribute, rsrc.FnGetAtt, 'hoge')
    self.assertEqual('The Referenced Attribute (test_network_gateway hoge) is incorrect.', str(error))
    self.mockclient.create_network_gateway.assert_called_once_with({'network_gateway': {'name': u'NetworkGateway', 'devices': [{'id': u'e52148ca-7db9-4ec3-abe6-2c7c0ff316eb', 'interface_name': u'breth1'}]}})
    self.mockclient.connect_network_gateway.assert_called_once_with(u'ed4c03b9-8251-4c09-acc4-e59ee9e6aa37', {'network_id': u'6af055d3-26f6-48dd-a597-7611d7e58d35', 'segmentation_id': 10, 'segmentation_type': u'vlan'})
    self.mockclient.show_network_gateway.assert_called_with(u'ed4c03b9-8251-4c09-acc4-e59ee9e6aa37')
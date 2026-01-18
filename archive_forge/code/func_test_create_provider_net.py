import copy
from unittest import mock
from neutronclient.common import exceptions as qe
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.resources.openstack.neutron import provider_net
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_create_provider_net(self):
    resource_type = 'networks'
    rsrc = self.create_provider_net()
    self.mockclient.show_network.side_effect = [stpnb, stpna, qe.NetworkNotFoundClient(status_code=404), stpna, qe.NetworkNotFoundClient(status_code=404)]
    self.mockclient.delete_network.side_effect = [None, qe.NetworkNotFoundClient(status_code=404)]
    rsrc.validate()
    scheduler.TaskRunner(rsrc.create)()
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    ref_id = rsrc.FnGetRefId()
    self.assertEqual('fc68ea2c-b60b-4b4f-bd82-94ec81110766', ref_id)
    self.assertIsNone(rsrc.FnGetAtt('status'))
    self.assertEqual('ACTIVE', rsrc.FnGetAtt('status'))
    self.assertRaises(exception.InvalidTemplateAttribute, rsrc.FnGetAtt, 'Foo')
    self.assertIsNone(scheduler.TaskRunner(rsrc.delete)())
    self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
    rsrc.state_set(rsrc.CREATE, rsrc.COMPLETE, 'to delete again')
    scheduler.TaskRunner(rsrc.delete)()
    self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
    self.mockclient.create_network.assert_called_once_with({'network': {'name': u'the_provider_network', 'admin_state_up': True, 'provider:network_type': 'vlan', 'provider:physical_network': 'physnet_1', 'provider:segmentation_id': '101', 'router:external': False, 'shared': True, 'availability_zone_hints': ['az1']}})
    self.mockclient.replace_tag.assert_called_with(resource_type, 'fc68ea2c-b60b-4b4f-bd82-94ec81110766', {'tags': ['tag1', 'tag2']})
    self.mockclient.show_network.assert_called_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766')
    self.assertEqual(5, self.mockclient.show_network.call_count)
    self.mockclient.delete_network.assert_called_with('fc68ea2c-b60b-4b4f-bd82-94ec81110766')
    self.assertEqual(2, self.mockclient.delete_network.call_count)
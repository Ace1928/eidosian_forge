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
def test_network_gatway_create_failed(self):
    self.mockclient.create_network_gateway.side_effect = qe.NeutronClientException
    self.stub_NetworkConstraint_validate()
    t = template_format.parse(gw_template)
    stack = utils.parse_stack(t)
    resource_defns = stack.t.resource_definitions(stack)
    rsrc = network_gateway.NetworkGateway('network_gateway', resource_defns['NetworkGateway'], stack)
    error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
    self.assertEqual('NeutronClientException: resources.network_gateway: An unknown exception occurred.', str(error))
    self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
    self.assertIsNone(scheduler.TaskRunner(rsrc.delete)())
    self.assertEqual((rsrc.DELETE, rsrc.COMPLETE), rsrc.state)
    self.mockclient.create_network_gateway.assert_called_once_with({'network_gateway': {'name': u'NetworkGateway', 'devices': [{'id': u'e52148ca-7db9-4ec3-abe6-2c7c0ff316eb', 'interface_name': u'breth1'}]}})
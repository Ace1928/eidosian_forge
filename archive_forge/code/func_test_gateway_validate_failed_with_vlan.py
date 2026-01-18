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
def test_gateway_validate_failed_with_vlan(self):
    t = template_format.parse(gw_template)
    del t['resources']['NetworkGateway']['properties']['connections'][0]['segmentation_id']
    stack = utils.parse_stack(t)
    resource_defns = stack.t.resource_definitions(stack)
    rsrc = network_gateway.NetworkGateway('test_network_gateway', resource_defns['NetworkGateway'], stack)
    self.stub_NetworkConstraint_validate()
    error = self.assertRaises(exception.StackValidationFailed, scheduler.TaskRunner(rsrc.validate))
    self.assertEqual('segmentation_id must be specified for using vlan', str(error))
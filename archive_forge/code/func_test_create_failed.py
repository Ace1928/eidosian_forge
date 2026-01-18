from unittest import mock
from neutronclient.common import exceptions
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.resources.openstack.neutron import metering
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_create_failed(self):
    self.mockclient.create_metering_label_rule.side_effect = exceptions.NeutronClientException()
    snippet = template_format.parse(metering_template)
    stack = utils.parse_stack(snippet)
    self.patchobject(stack['label'], 'FnGetRefId', return_value='1234')
    resource_defns = stack.t.resource_definitions(stack)
    rsrc = metering.MeteringRule('rule', resource_defns['rule'], stack)
    error = self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.create))
    self.assertEqual('NeutronClientException: resources.rule: An unknown exception occurred.', str(error))
    self.assertEqual((rsrc.CREATE, rsrc.FAILED), rsrc.state)
    self.mockclient.create_metering_label_rule.assert_called_once_with({'metering_label_rule': {'metering_label_id': '1234', 'remote_ip_prefix': '10.0.3.0/24', 'direction': 'ingress', 'excluded': False}})
import copy
from unittest import mock
import uuid
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients import progress
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine.resources import scheduler_hints as sh
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_instance_update_network_interfaces_new_old_all_different(self):
    """Tests updating NetworkInterfaces when new and old are different.

        Instance.handle_update supports changing the NetworkInterfaces,
        and makes the change making a resize API call against Nova.
        """
    return_server = self.fc.servers.list()[1]
    return_server.id = '1234'
    instance = self._create_test_instance(return_server, 'ud_network_interfaces')
    self._stub_glance_for_update()
    old_interfaces = [{'NetworkInterfaceId': 'ea29f957-cd35-4364-98fb-57ce9732c10d', 'DeviceIndex': '2'}]
    new_interfaces = [{'NetworkInterfaceId': '34b752ec-14de-416a-8722-9531015e04a5', 'DeviceIndex': '3'}, {'NetworkInterfaceId': 'd1e9c73c-04fe-4e9e-983c-d5ef94cd1a46', 'DeviceIndex': '1'}]
    before_props = self.instance_props.copy()
    before_props['NetworkInterfaces'] = old_interfaces
    update_props = self.instance_props.copy()
    update_props['NetworkInterfaces'] = new_interfaces
    after = instance.t.freeze(properties=update_props)
    before = instance.t.freeze(properties=before_props)
    self.fc.servers.get = mock.Mock(return_value=return_server)
    return_server.interface_detach = mock.Mock(return_value=None)
    return_server.interface_attach = mock.Mock(return_value=None)
    scheduler.TaskRunner(instance.update, after, before)()
    self.assertEqual((instance.UPDATE, instance.COMPLETE), instance.state)
    self.fc.servers.get.assert_called_with('1234')
    return_server.interface_detach.assert_called_once_with('ea29f957-cd35-4364-98fb-57ce9732c10d')
    return_server.interface_attach.assert_has_calls([mock.call('d1e9c73c-04fe-4e9e-983c-d5ef94cd1a46', None, None), mock.call('34b752ec-14de-416a-8722-9531015e04a5', None, None)], any_order=True)
    self.assertEqual(2, return_server.interface_attach.call_count)
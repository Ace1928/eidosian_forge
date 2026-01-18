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
def test_instance_create_with_volumes(self):
    return_server = self.fc.servers.list()[1]
    self.stub_VolumeConstraint_validate()
    instance = self._setup_test_instance(return_server, 'with_volumes', volumes=True)
    self.fc.servers.get = mock.Mock(return_value=return_server)
    attach_mock = self.patchobject(nova.NovaClientPlugin, 'attach_volume', side_effect=['cccc', 'dddd'])
    check_attach_mock = self.patchobject(cinder.CinderClientPlugin, 'check_attach_volume_complete', side_effect=[False, True, False, True])
    scheduler.TaskRunner(instance.create)()
    self.assertEqual((instance.CREATE, instance.COMPLETE), instance.state)
    self.assertEqual(2, attach_mock.call_count)
    attach_mock.assert_has_calls([mock.call(instance.resource_id, 'cccc', '/dev/vdc'), mock.call(instance.resource_id, 'dddd', '/dev/vdd')])
    self.assertEqual(4, check_attach_mock.call_count)
    check_attach_mock.assert_has_calls([mock.call('cccc'), mock.call('cccc'), mock.call('dddd'), mock.call('dddd')])
    bdm = {'vdb': '9ef5496e-7426-446a-bbc8-01f84d9c9972:snap::True'}
    self.mock_create.assert_called_once_with(image=1, flavor=1, key_name='test', name=utils.PhysName(self.stack.name, instance.name, limit=instance.physical_resource_name_limit), security_groups=None, userdata=mock.ANY, scheduler_hints={'foo': ['spam', 'ham', 'baz'], 'bar': 'eggs'}, meta=None, nics=None, availability_zone=None, block_device_mapping=bdm)
    self.fc.servers.get.assert_called_with(return_server.id)
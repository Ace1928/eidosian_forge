import collections
import copy
import json
from unittest import mock
from cinderclient import exceptions as cinder_exp
from novaclient import exceptions as nova_exp
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine.resources.openstack.cinder import volume as c_vol
from heat.engine.resources import scheduler_hints as sh
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.objects import resource_data as resource_data_object
from heat.tests.openstack.cinder import test_volume_utils as vt_base
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_cinder_volume_attachment_update_device(self):
    self.stack_name = 'test_cvolume_attach_udev_stack'
    m_v = self._mock_create_server_volume_script(vt_base.FakeVolume('attaching'))
    m_v2 = self._mock_create_server_volume_script(vt_base.FakeVolume('attaching'), update=True)
    fva = vt_base.FakeVolume('in-use')
    self._mock_create_volume(vt_base.FakeVolume('creating'), self.stack_name, extra_get_mocks=[m_v, fva, vt_base.FakeVolume('available'), m_v2])
    self.stub_VolumeConstraint_validate()
    self.fc.volumes.get_server_volume.side_effect = [fva, fva, fakes_nova.fake_exception()]
    self.fc.volumes.delete_server_volume.return_value = None
    stack = utils.parse_stack(self.t, stack_name=self.stack_name)
    self.create_volume(self.t, stack, 'volume')
    rsrc = self.create_attachment(self.t, stack, 'attachment')
    props = copy.deepcopy(rsrc.properties.data)
    props['mountpoint'] = ''
    props['volume_id'] = 'vol-123'
    after = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    scheduler.TaskRunner(rsrc.update, after)()
    self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
    self.fc.volumes.get_server_volume.assert_called_with(u'WikiDatabase', 'vol-123')
    self.fc.volumes.delete_server_volume.assert_called_with('WikiDatabase', 'vol-123')
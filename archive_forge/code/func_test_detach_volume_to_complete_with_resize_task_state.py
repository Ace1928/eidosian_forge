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
def test_detach_volume_to_complete_with_resize_task_state(self):
    fv = vt_base.FakeVolume('creating')
    self.stack_name = 'test_cvolume_detach_with_resize_task_state_stack'
    self.stub_SnapshotConstraint_validate()
    self.stub_VolumeConstraint_validate()
    self.stub_VolumeTypeConstraint_validate()
    self.cinder_fc.volumes.create.return_value = fv
    fv_ready = vt_base.FakeVolume('available', id=fv.id)
    self.cinder_fc.volumes.get.side_effect = [fv, fv_ready]
    self.fc.volumes.delete_server_volume.side_effect = [nova_exp.Conflict('409')]
    self.t['resources']['volume']['properties'].update({'volume_type': 'lvm'})
    stack = utils.parse_stack(self.t, stack_name=self.stack_name)
    rsrc = self.create_volume(self.t, stack, 'volume')
    prg_detach = mock.MagicMock(called=False, srv_id='InstanceInResize')
    self.assertEqual(False, rsrc._detach_volume_to_complete(prg_detach))
    self.assertEqual(False, prg_detach.called)
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
def test_cinder_create_with_read_only(self):
    fv = vt_base.FakeVolume('with_read_only_access_mode')
    self.stack_name = 'test_create_with_read_only'
    self.cinder_fc.volumes.create.return_value = fv
    update_readonly_mock = self.patchobject(self.cinder_fc.volumes, 'update_readonly_flag', return_value=None)
    fv_ready = vt_base.FakeVolume('available', id=fv.id)
    self.cinder_fc.volumes.get.return_value = fv_ready
    self.t['resources']['volume']['properties'] = {'size': '1', 'name': 'ImageVolume', 'description': 'ImageVolumeDescription', 'availability_zone': 'nova', 'read_only': False}
    stack = utils.parse_stack(self.t, stack_name=self.stack_name)
    self.create_volume(self.t, stack, 'volume')
    update_readonly_mock.assert_called_once_with(fv.id, False)
    self.cinder_fc.volumes.create.assert_called_once_with(size=1, availability_zone='nova', description='ImageVolumeDescription', name='ImageVolume', metadata={})
    self.cinder_fc.volumes.get.assert_called_once_with(fv.id)
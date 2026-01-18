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
def test_cinder_create_with_scheduler_hints(self):
    fv = vt_base.FakeVolume('creating')
    self.cinder_fc.volumes.create.return_value = fv
    fv_ready = vt_base.FakeVolume('available', id=fv.id)
    self.cinder_fc.volumes.get.side_effect = [fv, fv_ready]
    self.stack_name = 'test_cvolume_scheduler_hints_stack'
    stack = utils.parse_stack(self.t, stack_name=self.stack_name)
    self.patchobject(stack['volume3'], '_store_config_default_properties')
    self.create_volume(self.t, stack, 'volume3')
    self.cinder_fc.volumes.create.assert_called_once_with(size=1, name='test_name', description=None, availability_zone='nova', scheduler_hints={'hint1': 'good_advice'}, metadata={})
    self.assertEqual(2, self.cinder_fc.volumes.get.call_count)
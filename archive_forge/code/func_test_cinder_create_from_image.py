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
def test_cinder_create_from_image(self):
    fv = vt_base.FakeVolume('downloading')
    self.stack_name = 'test_cvolume_create_from_img_stack'
    image_id = '46988116-6703-4623-9dbc-2bc6d284021b'
    self.patchobject(glance.GlanceClientPlugin, 'find_image_by_name_or_id', return_value=image_id)
    self.cinder_fc.volumes.create.return_value = fv
    fv_ready = vt_base.FakeVolume('available', id=fv.id)
    self.cinder_fc.volumes.get.side_effect = [fv, fv_ready]
    self.t['resources']['volume']['properties'] = {'size': '1', 'name': 'ImageVolume', 'description': 'ImageVolumeDescription', 'availability_zone': 'nova', 'image': image_id}
    stack = utils.parse_stack(self.t, stack_name=self.stack_name)
    self.create_volume(self.t, stack, 'volume')
    glance.GlanceClientPlugin.find_image_by_name_or_id.assert_called_with(image_id)
    self.cinder_fc.volumes.create.assert_called_once_with(size=1, availability_zone='nova', description='ImageVolumeDescription', name='ImageVolume', imageRef=image_id, metadata={})
    self.assertEqual(2, self.cinder_fc.volumes.get.call_count)
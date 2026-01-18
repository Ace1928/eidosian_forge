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
def test_cinder_volume_retype(self):
    fv = vt_base.FakeVolume('available', size=1, name='my_vol', description='test')
    self.stack_name = 'test_cvolume_retype'
    new_vol_type = 'new_type'
    self.patchobject(cinder.CinderClientPlugin, '_create', return_value=self.cinder_fc)
    self.cinder_fc.volumes.create.return_value = fv
    self.cinder_fc.volumes.get.return_value = fv
    stack = utils.parse_stack(self.t, stack_name=self.stack_name)
    rsrc = self.create_volume(self.t, stack, 'volume2')
    props = copy.deepcopy(rsrc.properties.data)
    props['volume_type'] = new_vol_type
    after = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    self.patchobject(cinder.CinderClientPlugin, 'get_volume_type', return_value=new_vol_type)
    self.patchobject(self.cinder_fc.volumes, 'retype')
    scheduler.TaskRunner(rsrc.update, after)()
    self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
    self.assertEqual(1, self.cinder_fc.volumes.retype.call_count)
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
def test_cinder_volume_update_no_need_replace(self):
    fv = vt_base.FakeVolume('creating')
    self.stack_name = 'test_update_no_need_replace'
    self.cinder_fc.volumes.create.return_value = fv
    vol_name = utils.PhysName(self.stack_name, 'volume2')
    fv_ready = vt_base.FakeVolume('available', id=fv.id, size=2, attachments=[])
    self.cinder_fc.volumes.get.return_value = fv_ready
    stack = utils.parse_stack(self.t, stack_name=self.stack_name)
    rsrc = self.create_volume(self.t, stack, 'volume2')
    props = copy.deepcopy(rsrc.properties.data)
    props['size'] = 1
    after = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    update_task = scheduler.TaskRunner(rsrc.update, after)
    ex = self.assertRaises(exception.ResourceFailure, update_task)
    self.assertEqual((rsrc.UPDATE, rsrc.FAILED), rsrc.state)
    self.assertIn('NotSupported: resources.volume2: Shrinking volume is not supported', str(ex))
    props = copy.deepcopy(rsrc.properties.data)
    props['size'] = 3
    after = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    scheduler.TaskRunner(rsrc.update, after)()
    self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
    self.cinder_fc.volumes.create.assert_called_once_with(size=2, availability_zone='nova', description=None, name=vol_name, metadata={})
    self.cinder_fc.volumes.extend.assert_called_once_with(fv.id, 3)
    self.cinder_fc.volumes.get.assert_called_with(fv.id)
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
def test_cinder_volume_extend_created_from_backup_with_same_size(self):
    self.stack_name = 'test_cvolume_extend_snapsht_stack'
    fv = vt_base.FakeVolume('available', size=2)
    self.stub_VolumeBackupConstraint_validate()
    fvbr = vt_base.FakeBackupRestore('vol-123')
    self.patchobject(self.cinder_fc.restores, 'restore', return_value=fvbr)
    self.cinder_fc.volumes.get.side_effect = [vt_base.FakeVolume('restoring-backup'), vt_base.FakeVolume('available'), fv]
    vol_name = utils.PhysName(self.stack_name, 'volume')
    self.cinder_fc.volumes.update.return_value = None
    self.t['resources']['volume']['properties'] = {'availability_zone': 'nova', 'backup_id': 'backup-123'}
    stack = utils.parse_stack(self.t, stack_name=self.stack_name)
    rsrc = self.create_volume(self.t, stack, 'volume')
    self.assertEqual((rsrc.CREATE, rsrc.COMPLETE), rsrc.state)
    self.assertEqual('available', fv.status)
    props = copy.deepcopy(rsrc.properties.data)
    props['size'] = 2
    after = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    update_task = scheduler.TaskRunner(rsrc.update, after)
    self.assertIsNone(update_task())
    self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
    self.cinder_fc.restores.restore.assert_called_once_with('backup-123')
    self.cinder_fc.volumes.update.assert_called_once_with('vol-123', description=None, name=vol_name)
    self.assertEqual(3, self.cinder_fc.volumes.get.call_count)
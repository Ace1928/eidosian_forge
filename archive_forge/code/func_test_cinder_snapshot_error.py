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
def test_cinder_snapshot_error(self):
    self.stack_name = 'test_cvolume_snpsht_err_stack'
    self.cinder_fc.volumes.create.return_value = vt_base.FakeVolume('creating')
    fv = vt_base.FakeVolume('available')
    self.cinder_fc.volumes.get.return_value = fv
    fb = vt_base.FakeBackup('creating')
    self.patchobject(self.cinder_fc.backups, 'create', return_value=fb)
    fail_reason = 'Could not determine which Swift endpoint to use'
    self.patchobject(self.cinder_fc.backups, 'get', return_value=vt_base.FakeBackup('error', fail_reason=fail_reason))
    t = template_format.parse(single_cinder_volume_template)
    stack = utils.parse_stack(t, stack_name=self.stack_name)
    rsrc = stack['volume']
    self.patchobject(rsrc, '_store_config_default_properties')
    scheduler.TaskRunner(rsrc.create)()
    self.assertRaises(exception.ResourceFailure, scheduler.TaskRunner(rsrc.snapshot))
    self.assertEqual((rsrc.SNAPSHOT, rsrc.FAILED), rsrc.state)
    self.assertIn(fail_reason, rsrc.status_reason)
    self.assertEqual({u'backup_id': u'backup-123'}, resource_data_object.ResourceData.get_all(rsrc))
    self.cinder_fc.volumes.create.assert_called_once_with(size=1, availability_zone=None, description='test_description', name='test_name', metadata={})
    self.cinder_fc.backups.create.assert_called_once_with(fv.id, force=True)
    self.cinder_fc.volumes.get.assert_called_once_with(fv.id)
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
def test_cinder_volume_update_name_and_metadata(self):
    fv = vt_base.FakeVolume('creating', size=1, name='my_vol', description='test')
    self.stack_name = 'test_cvolume_updname_stack'
    update_name = 'update_name'
    meta = {'Key': 'New Value'}
    update_description = 'update_description'
    kwargs = {'name': update_name, 'description': update_description}
    fv_ready = vt_base.FakeVolume('available', id=fv.id)
    self._mock_create_volume(fv, self.stack_name, extra_get_mocks=[fv_ready])
    self.cinder_fc.volumes.update.return_value = None
    self.cinder_fc.volumes.update_all_metadata.return_value = None
    stack = utils.parse_stack(self.t, stack_name=self.stack_name)
    rsrc = self.create_volume(self.t, stack, 'volume')
    props = copy.deepcopy(rsrc.properties.data)
    props['name'] = update_name
    props['description'] = update_description
    props['metadata'] = meta
    after = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    scheduler.TaskRunner(rsrc.update, after)()
    self.assertEqual((rsrc.UPDATE, rsrc.COMPLETE), rsrc.state)
    self.cinder_fc.volumes.update.assert_called_once_with(fv_ready, **kwargs)
    self.cinder_fc.volumes.update_all_metadata.assert_called_once_with(fv_ready, meta)
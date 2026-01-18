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
def test_handle_delete_snapshot_no_backup(self):
    self.stack_name = 'test_handle_delete_snapshot_no_backup'
    mock_vs = {'resource_data': {}}
    t = template_format.parse(single_cinder_volume_template)
    stack = utils.parse_stack(t, stack_name=self.stack_name)
    rsrc = c_vol.CinderVolume('volume', stack.t.resource_definitions(stack)['volume'], stack)
    self.assertIsNone(rsrc.handle_delete_snapshot(mock_vs))
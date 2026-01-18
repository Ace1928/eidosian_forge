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
def test_cinder_create_with_size_snapshot_and_sourcevol(self):
    self.stack_name = 'test_create_with_size_snapshot_and_sourcevol'
    combinations = {'size': 1, 'source_volid': 'volume-123', 'snapshot_id': 'snapshot-123'}
    self.stub_VolumeConstraint_validate()
    self.stub_SnapshotConstraint_validate()
    err_msg = 'If "size" is provided, only one of "image", "imageRef", "source_volid", "snapshot_id" can be specified, but currently specified options: [\'snapshot_id\', \'source_volid\'].'
    self._test_cinder_create_invalid_property_combinations(self.stack_name, combinations, err_msg, exception.StackValidationFailed)
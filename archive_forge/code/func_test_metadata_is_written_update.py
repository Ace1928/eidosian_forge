import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_metadata_is_written_update(self):
    nowish = timeutils.utcnow()
    reason = 'cool as'
    prev_cooldown_end = nowish + datetime.timedelta(seconds=100)
    previous_meta = {'cooldown_end': {prev_cooldown_end.isoformat(): 'ChangeInCapacity : 1'}}
    self.patchobject(self.group, 'metadata_get', return_value=previous_meta)
    meta_set = self.patchobject(self.group, 'metadata_set')
    self.patchobject(timeutils, 'utcnow', return_value=nowish)
    self.group._finished_scaling(60, reason)
    meta_set.assert_called_once_with({'cooldown_end': {prev_cooldown_end.isoformat(): reason}, 'scaling_in_progress': False})
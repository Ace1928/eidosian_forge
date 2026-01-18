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
def test_cooldown_is_in_progress_toosoon(self):
    cooldown_end = timeutils.utcnow() + datetime.timedelta(seconds=60)
    previous_meta = {'cooldown_end': {cooldown_end.isoformat(): 'ChangeInCapacity : 1'}}
    self.patchobject(self.group, 'metadata_get', return_value=previous_meta)
    self.assertRaises(resource.NoActionRequired, self.group._check_scaling_allowed, 60)
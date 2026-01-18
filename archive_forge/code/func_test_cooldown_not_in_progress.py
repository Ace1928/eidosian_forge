import datetime
import json
from unittest import mock
from oslo_utils import timeutils
from heat.common import exception
from heat.common import grouputils
from heat.common import template_format
from heat.engine import resource
from heat.engine import rsrc_defn
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_cooldown_not_in_progress(self):
    awhile_after = timeutils.utcnow() + datetime.timedelta(seconds=60)
    previous_meta = {'cooldown_end': {awhile_after.isoformat(): 'change_in_capacity : 1'}, 'scaling_in_progress': False}
    timeutils.set_time_override()
    timeutils.advance_time_seconds(100)
    self.patchobject(self.group, 'metadata_get', return_value=previous_meta)
    self.assertIsNone(self.group._check_scaling_allowed(60))
    timeutils.clear_time_override()
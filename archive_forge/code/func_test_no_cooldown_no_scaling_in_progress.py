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
def test_no_cooldown_no_scaling_in_progress(self):
    awhile_ago = timeutils.utcnow() - datetime.timedelta(seconds=100)
    previous_meta = {'scaling_in_progress': False, awhile_ago.isoformat(): 'change_in_capacity : 1'}
    self.patchobject(self.group, 'metadata_get', return_value=previous_meta)
    self.assertIsNone(self.group._check_scaling_allowed(60))
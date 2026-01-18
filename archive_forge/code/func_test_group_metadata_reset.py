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
def test_group_metadata_reset(self):
    self.group.state_set('CREATE', 'COMPLETE')
    metadata = {'scaling_in_progress': True}
    self.group.metadata_set(metadata)
    self.group.handle_metadata_reset()
    new_metadata = self.group.metadata_get()
    self.assertEqual({'scaling_in_progress': False}, new_metadata)
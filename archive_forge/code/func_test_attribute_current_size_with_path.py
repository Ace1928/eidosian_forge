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
def test_attribute_current_size_with_path(self):
    mock_instances = self.patchobject(grouputils, 'get_size')
    mock_instances.return_value = 4
    self.assertEqual(4, self.group.FnGetAtt('current_size', 'name'))
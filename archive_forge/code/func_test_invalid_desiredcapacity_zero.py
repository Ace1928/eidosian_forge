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
def test_invalid_desiredcapacity_zero(self):
    t = template_format.parse(as_template)
    properties = t['Resources']['WebServerGroup']['Properties']
    properties['MinSize'] = '1'
    properties['MaxSize'] = '3'
    properties['DesiredCapacity'] = '0'
    stack = utils.parse_stack(t, params=inline_templates.as_params)
    self.stub_SnapshotConstraint_validate()
    self.stub_ImageConstraint_validate()
    self.stub_FlavorConstraint_validate()
    e = self.assertRaises(exception.StackValidationFailed, self.validate_scaling_group, t, stack, 'WebServerGroup')
    expected_msg = 'DesiredCapacity must be between MinSize and MaxSize'
    self.assertEqual(expected_msg, str(e))
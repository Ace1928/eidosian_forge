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
def test_initial_size(self):
    t = template_format.parse(as_template)
    properties = t['Resources']['WebServerGroup']['Properties']
    properties['MinSize'] = self.mins
    properties['MaxSize'] = self.maxs
    properties['DesiredCapacity'] = self.desired
    stack = utils.parse_stack(t, params=inline_templates.as_params)
    group = stack['WebServerGroup']
    with mock.patch.object(group, '_create_template') as mock_cre_temp:
        group.child_template()
        mock_cre_temp.assert_called_once_with(self.expected)
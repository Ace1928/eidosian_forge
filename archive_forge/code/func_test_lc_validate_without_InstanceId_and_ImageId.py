from unittest import mock
from heat.common import exception
from heat.common import short_id
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine import scheduler
from heat.tests.autoscaling import inline_templates
from heat.tests import common
from heat.tests import utils
def test_lc_validate_without_InstanceId_and_ImageId(self):
    t = template_format.parse(inline_templates.as_template)
    lcp = t['Resources']['LaunchConfig']['Properties']
    lcp.pop('ImageId')
    stack = utils.parse_stack(t, inline_templates.as_params)
    rsrc = stack['LaunchConfig']
    self.stub_SnapshotConstraint_validate()
    self.stub_FlavorConstraint_validate()
    e = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
    ex_msg = 'If without InstanceId, ImageId and InstanceType are required.'
    self.assertIn(ex_msg, str(e))
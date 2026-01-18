import collections
from unittest import mock
from heatclient import exc
from heatclient.osc.v1 import stack_failures
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
def test_build_failed_resources_not_found(self):
    self.resource_client.list.side_effect = [[self.failed_template_resource, self.other_failed_template_resource, self.working_resource], exc.HTTPNotFound(), []]
    failures = self.cmd._build_failed_resources('stack')
    expected = collections.OrderedDict()
    expected['stack.my_templateresource'] = self.failed_template_resource
    expected['stack.my_othertemplateresource'] = self.other_failed_template_resource
    self.assertEqual(expected, failures)
import collections
from unittest import mock
from heatclient import exc
from heatclient.osc.v1 import stack_failures
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
def test_build_software_deployments_not_found(self):
    resources = {'stack.my_server': self.working_resource, 'stack.my_deployment': self.failed_deployment_resource}
    self.software_deployments_client.get.side_effect = exc.HTTPNotFound()
    deployments = self.cmd._build_software_deployments(resources)
    self.assertEqual({}, deployments)
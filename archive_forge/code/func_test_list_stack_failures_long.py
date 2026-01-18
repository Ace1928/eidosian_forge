import collections
from unittest import mock
from heatclient import exc
from heatclient.osc.v1 import stack_failures
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
def test_list_stack_failures_long(self):
    self.resource_client.list.side_effect = [[self.failed_template_resource, self.other_failed_template_resource, self.working_resource, self.failed_deployment_resource], [self.failed_resource], []]
    arglist = ['--long', 'stack']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.cmd.take_action(parsed_args)
    self.assertEqual(self.app.stdout.make_string(), 'stack.my_templateresource.my_server:\n  resource_type: OS::Nova::Server\n  physical_resource_id: cccc\n  status: CREATE_FAILED\n  status_reason: |\n    All gone Pete Tong\nstack.my_othertemplateresource:\n  resource_type: My::OtherTemplateResource\n  physical_resource_id: dddd\n  status: CREATE_FAILED\n  status_reason: |\n    RPC timeout\nstack.my_deployment:\n  resource_type: OS::Heat::SoftwareDeployment\n  physical_resource_id: eeee\n  status: CREATE_FAILED\n  status_reason: |\n    Returned deploy_statuscode 1\n  deploy_stdout: |\n    1\n    2\n    3\n    4\n    5\n    6\n    7\n    8\n    9\n    10\n    11\n    12\n  deploy_stderr: |\n    It broke\n')
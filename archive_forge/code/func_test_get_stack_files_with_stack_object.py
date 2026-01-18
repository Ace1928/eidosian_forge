from unittest import mock
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.orchestration.v1 import _proxy
from openstack.orchestration.v1 import resource
from openstack.orchestration.v1 import software_config as sc
from openstack.orchestration.v1 import software_deployment as sd
from openstack.orchestration.v1 import stack
from openstack.orchestration.v1 import stack_environment
from openstack.orchestration.v1 import stack_event
from openstack.orchestration.v1 import stack_files
from openstack.orchestration.v1 import stack_template
from openstack.orchestration.v1 import template
from openstack import proxy
from openstack.tests.unit import test_proxy_base
@mock.patch.object(stack_files.StackFiles, 'fetch')
def test_get_stack_files_with_stack_object(self, mock_fetch):
    stack_id = '1234'
    stack_name = 'test_stack'
    stk = stack.Stack(id=stack_id, name=stack_name)
    mock_fetch.return_value = {'file': 'content'}
    res = self.proxy.get_stack_files(stk)
    self.assertEqual({'file': 'content'}, res)
    mock_fetch.assert_called_once_with(self.proxy)
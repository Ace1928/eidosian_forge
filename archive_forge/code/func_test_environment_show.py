from unittest import mock
import testscenarios
from testscenarios import scenarios as scnrs
import testtools
from heatclient.v1 import stacks
def test_environment_show(self):
    manager = mock.MagicMock()
    stack = mock_stack(manager, 'env_stack', 'env1')
    stack.environment()
    manager.environment.assert_called_once_with('env_stack/env1')
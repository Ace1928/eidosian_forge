import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_deployment
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
from heatclient.v1 import software_deployments
@mock.patch('heatclient.common.deployment_utils.build_signal_id', return_value='signal_id')
def test_deployment_create(self, mock_build):
    arglist = ['my_deploy', '--server', self.server_id]
    expected_cols = ('action', 'config_id', 'id', 'input_values', 'server_id', 'signal_id', 'status', 'status_reason')
    expected_data = ('UPDATE', self.config_id, self.deploy_id, {}, self.server_id, 'signal_id', 'IN_PROGRESS', None)
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.config_client.create.assert_called_with(**self.config_defaults)
    self.sd_client.create.assert_called_with(**self.deploy_defaults)
    self.assertEqual(expected_cols, columns)
    self.assertEqual(expected_data, data)
import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_deployment
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
from heatclient.v1 import software_deployments
def test_deployment_show_long(self):
    arglist = ['my_deployment', '--long']
    cols = ['id', 'server_id', 'config_id', 'creation_time', 'updated_time', 'status', 'status_reason', 'input_values', 'action', 'output_values']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.sd_client.get.return_value = software_deployments.SoftwareDeployment(None, self.get_response)
    columns, data = self.cmd.take_action(parsed_args)
    self.sd_client.get.assert_called_once_with(**{'deployment_id': 'my_deployment'})
    self.assertEqual(cols, columns)
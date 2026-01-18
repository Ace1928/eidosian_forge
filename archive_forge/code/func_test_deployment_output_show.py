import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_deployment
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
from heatclient.v1 import software_deployments
def test_deployment_output_show(self):
    arglist = ['85c3a507-351b-4b28-a7d8-531c8d53f4e6', '--all', '--long']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.sd_client.get.return_value = software_deployments.SoftwareDeployment(None, self.get_response)
    self.cmd.take_action(parsed_args)
    self.sd_client.get.assert_called_with(**{'deployment_id': '85c3a507-351b-4b28-a7d8-531c8d53f4e6'})
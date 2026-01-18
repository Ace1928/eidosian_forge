import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_deployment
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
from heatclient.v1 import software_deployments
def test_deployment_delete_success(self):
    arglist = ['test_deployment']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.cmd.take_action(parsed_args)
    self.sd_client.delete.assert_called_with(deployment_id='test_deployment')
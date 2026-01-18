import copy
from unittest import mock
from osc_lib import exceptions as exc
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_deployment
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
from heatclient.v1 import software_deployments
def test_deployment_output_show_not_found(self):
    arglist = ['85c3a507-351b-4b28-a7d8-531c8d53f4e6', '--all']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.sd_client.get.side_effect = heat_exc.HTTPNotFound()
    self.assertRaises(exc.CommandError, self.cmd.take_action, parsed_args)
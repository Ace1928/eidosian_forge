from unittest import mock
from osc_lib import exceptions as exc
import yaml
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_config
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
def test_config_list_marker(self):
    arglist = ['--marker', 'id123']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.cmd.take_action(parsed_args)
    self.mock_client.software_configs.list.assert_called_with(marker='id123')
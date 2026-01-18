from unittest import mock
from osc_lib import exceptions as exc
import yaml
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_config
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
def test_config_show_config_only(self):
    arglist = ['--config-only', '96dfee3f-27b7-42ae-a03e-966226871ae6']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.mock_client.software_configs.get.assert_called_with(**{'config_id': '96dfee3f-27b7-42ae-a03e-966226871ae6'})
    self.assertIsNone(columns)
    self.assertIsNone(data)
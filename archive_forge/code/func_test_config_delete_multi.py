from unittest import mock
from osc_lib import exceptions as exc
import yaml
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_config
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
def test_config_delete_multi(self):
    arglist = ['id_123', 'id_456']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    self.cmd.take_action(parsed_args)
    self.mock_delete.assert_has_calls([mock.call(config_id='id_123'), mock.call(config_id='id_456')])
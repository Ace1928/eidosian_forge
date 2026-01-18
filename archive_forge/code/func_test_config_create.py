from unittest import mock
from osc_lib import exceptions as exc
import yaml
from heatclient import exc as heat_exc
from heatclient.osc.v1 import software_config
from heatclient.tests.unit.osc.v1 import fakes as orchestration_fakes
from heatclient.v1 import software_configs
def test_config_create(self):
    properties = {'config': '', 'group': 'Heat::Ungrouped', 'name': 'test', 'options': {}, 'inputs': [], 'outputs': []}
    arglist = ['test']
    parsed_args = self.check_parser(self.cmd, arglist, [])
    columns, rows = self.cmd.take_action(parsed_args)
    self.mock_client.stacks.validate.assert_called_with(**{'template': {'heat_template_version': '2013-05-23', 'resources': {'test': {'type': 'OS::Heat::SoftwareConfig', 'properties': properties}}}})
    self.mock_client.software_configs.create.assert_called_with(**properties)
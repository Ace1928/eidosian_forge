import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_connector as bm_vol_connector
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_connector_set_multiple_extras(self):
    arglist = [baremetal_fakes.baremetal_volume_connector_uuid, '--extra', 'key1=val1', '--extra', 'key2=val2']
    verifylist = [('volume_connector', baremetal_fakes.baremetal_volume_connector_uuid), ('extra', ['key1=val1', 'key2=val2'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.volume_connector.update.assert_called_once_with(baremetal_fakes.baremetal_volume_connector_uuid, [{'path': '/extra/key1', 'value': 'val1', 'op': 'add'}, {'path': '/extra/key2', 'value': 'val2', 'op': 'add'}])
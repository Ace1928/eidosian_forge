import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_connector as bm_vol_connector
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_connector_unset_no_property(self):
    arglist = [baremetal_fakes.baremetal_volume_connector_uuid]
    verifylist = [('volume_connector', baremetal_fakes.baremetal_volume_connector_uuid)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.baremetal_mock.volume_connector.update.assert_not_called()
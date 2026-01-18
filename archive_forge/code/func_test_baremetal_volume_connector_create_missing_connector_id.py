import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_connector as bm_vol_connector
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_connector_create_missing_connector_id(self):
    arglist = ['--node', baremetal_fakes.baremetal_uuid, '--type', baremetal_fakes.baremetal_volume_connector_type, '--uuid', baremetal_fakes.baremetal_volume_connector_uuid]
    verifylist = None
    self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
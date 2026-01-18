import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_volume_target as bm_vol_target
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_volume_target_set_invalid_boot_index(self):
    new_boot_idx = 'string'
    arglist = [baremetal_fakes.baremetal_volume_target_uuid, '--boot-index', new_boot_idx]
    verifylist = None
    self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
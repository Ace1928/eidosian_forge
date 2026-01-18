import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_driver
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_driver_list_invalid_fields(self):
    arglist = ['--fields', 'name', 'invalid']
    verifylist = [('fields', [['name', 'invalid']])]
    self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
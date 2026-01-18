import copy
from osc_lib.tests import utils as oscutils
from ironicclient.osc.v1 import baremetal_conductor
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_conductor_show_invalid_fields(self):
    arglist = ['xxxxx', '--fields', 'hostname', 'invalid']
    verifylist = [('conductor', 'xxxxx'), ('fields', [['hostname', 'invalid']])]
    self.assertRaises(oscutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
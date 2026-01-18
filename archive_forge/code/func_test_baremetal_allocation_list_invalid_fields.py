import copy
from unittest import mock
from osc_lib.tests import utils as osctestutils
from ironicclient import exc
from ironicclient.osc.v1 import baremetal_allocation
from ironicclient.tests.unit.osc.v1 import fakes as baremetal_fakes
def test_baremetal_allocation_list_invalid_fields(self):
    arglist = ['--fields', 'uuid', 'invalid']
    verifylist = [('fields', [['uuid', 'invalid']])]
    self.assertRaises(osctestutils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
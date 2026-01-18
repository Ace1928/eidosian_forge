import re
from unittest import mock
from osc_lib import exceptions
from osc_lib.tests import utils
import testtools
from neutronclient.osc import utils as osc_utils
from neutronclient.osc.v2.fwaas import firewallrule
from neutronclient.tests.unit.osc.v2 import fakes as test_fakes
from neutronclient.tests.unit.osc.v2.fwaas import common
from neutronclient.tests.unit.osc.v2.fwaas import fakes
def test_create_with_src_fwg_and_no(self):
    target = self.resource['id']
    fwg = 'my-fwg'
    arglist = [target, '--source-firewall-group', fwg, '--no-source-firewall-group']
    verifylist = [(self.res, target), ('source_firewall_group', fwg), ('no_source_firewall_group', True)]
    self.assertRaises(utils.ParserException, self.check_parser, self.cmd, arglist, verifylist)
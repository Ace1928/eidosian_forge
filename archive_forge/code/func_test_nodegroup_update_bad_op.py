import copy
from unittest import mock
from unittest.mock import call
from magnumclient.osc.v1 import nodegroups as osc_nodegroups
from magnumclient.tests.osc.unit.v1 import fakes as magnum_fakes
def test_nodegroup_update_bad_op(self):
    arglist = ['cluster', 'ng1', 'foo', 'bar']
    verifylist = [('cluster', 'cluster'), ('nodegroup', 'ng1'), ('op', 'foo'), ('attributes', ['bar'])]
    self.assertRaises(magnum_fakes.MagnumParseException, self.check_parser, self.cmd, arglist, verifylist)
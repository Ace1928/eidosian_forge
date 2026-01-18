from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_flow_classifier
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_create_flow_classifier_default_options(self):
    arglist = ['--logical-source-port', self._fc['logical_source_port'], '--ethertype', self._fc['ethertype'], self._fc['name']]
    verifylist = [('logical_source_port', self._fc['logical_source_port']), ('ethertype', self._fc['ethertype']), ('name', self._fc['name'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network.create_sfc_flow_classifier.assert_called_once_with(**{'name': self._fc['name'], 'logical_source_port': self._fc['logical_source_port'], 'ethertype': self._fc['ethertype']})
    self.assertEqual(self.columns, columns)
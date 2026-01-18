from unittest import mock
from osc_lib import exceptions
import testtools
from neutronclient.osc.v2.sfc import sfc_flow_classifier
from neutronclient.tests.unit.osc.v2.sfc import fakes
def test_delete_multiple_flow_classifiers_with_exception(self):
    client = self.app.client_manager.network
    target1 = self._flow_classifier[0]['id']
    arglist = [target1]
    verifylist = [('flow_classifier', [target1])]
    client.find_sfc_flow_classifier.side_effect = [target1, exceptions.CommandError]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    msg = '1 of 2 flow classifier(s) failed to delete.'
    with testtools.ExpectedException(exceptions.CommandError) as e:
        self.cmd.take_action(parsed_args)
        self.assertEqual(msg, str(e))
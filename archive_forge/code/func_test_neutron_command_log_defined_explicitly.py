import logging
import testtools
from neutronclient.neutron import v2_0 as neutronV20
def test_neutron_command_log_defined_explicitly(self):

    class FakeCommand(neutronV20.NeutronCommand):
        log = None
    self.assertTrue(hasattr(FakeCommand, 'log'))
    self.assertIsNone(FakeCommand.log)
import logging
import testtools
from neutronclient.neutron import v2_0 as neutronV20
def test_neutron_command_meta_defines_log(self):

    class FakeCommand(neutronV20.NeutronCommand):
        pass
    self.assertTrue(hasattr(FakeCommand, 'log'))
    self.assertIsInstance(FakeCommand.log, logging.getLoggerClass())
    self.assertEqual(__name__ + '.FakeCommand', FakeCommand.log.name)
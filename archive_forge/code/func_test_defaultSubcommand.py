from twisted.python import usage
from twisted.trial import unittest
def test_defaultSubcommand(self):
    """
        Flags and options in the default subcommand are assigned.
        """
    o = SubCommandOptions()
    o.defaultSubCommand = 'inquest'
    o.parseOptions(['--europian-swallow'])
    self.assertTrue(o['europian-swallow'])
    self.assertEqual(o.subCommand, 'inquisition')
    self.assertIsInstance(o.subOptions, InquisitionOptions)
    self.assertFalse(o.subOptions['expect'])
    self.assertEqual(o.subOptions['torture-device'], 'comfy-chair')
import os
from breezy.tests import TestCaseWithTransport
def assertInventoryEqual(self, expected, args=None, **kwargs):
    """Test that the output of 'brz inventory' is as expected.

        Any arguments supplied will be passed to run_bzr.
        """
    command = 'inventory'
    if args is not None:
        command += ' ' + args
    out, err = self.run_bzr(command, **kwargs)
    self.assertEqual(expected, out)
    self.assertEqual('', err)
import sys
from twisted.internet._glibbase import ensureNotImported
from twisted.trial.unittest import TestCase
def test_ensureWhenFailedToImport(self):
    """
        If the specified modules have been set to L{None} in C{sys.modules},
        L{ensureNotImported} does not complain.
        """
    modules = {'m2': None}
    self.patch(sys, 'modules', modules)
    ensureNotImported(['m1', 'm2'], 'A message.', preventImports=['m1', 'm2'])
    self.assertEqual(modules, {'m1': None, 'm2': None})
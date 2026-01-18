import sys
from twisted.internet._glibbase import ensureNotImported
from twisted.trial.unittest import TestCase
def test_ensureFailsWhenImported(self):
    """
        If one of the specified modules has been previously imported,
        L{ensureNotImported} raises an exception.
        """
    module = object()
    modules = {'m2': module}
    self.patch(sys, 'modules', modules)
    e = self.assertRaises(ImportError, ensureNotImported, ['m1', 'm2'], 'A message.', preventImports=['m1', 'm2'])
    self.assertEqual(modules, {'m2': module})
    self.assertEqual(e.args, ('A message.',))
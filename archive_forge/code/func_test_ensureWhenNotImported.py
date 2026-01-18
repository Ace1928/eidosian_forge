import sys
from twisted.internet._glibbase import ensureNotImported
from twisted.trial.unittest import TestCase
def test_ensureWhenNotImported(self):
    """
        If the specified modules have never been imported, and import
        prevention is requested, L{ensureNotImported} makes sure they will not
        be imported in the future.
        """
    modules = {}
    self.patch(sys, 'modules', modules)
    ensureNotImported(['m1', 'm2'], 'A message.', preventImports=['m1', 'm2', 'm3'])
    self.assertEqual(modules, {'m1': None, 'm2': None, 'm3': None})
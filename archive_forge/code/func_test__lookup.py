import sys
from breezy import tests
from breezy.tests import features
def test__lookup(self):
    obj = self.module.SimpleSet()
    self.assertLookup(643, '<null>', obj, _Hashable(643))
    self.assertLookup(643, '<null>', obj, _Hashable(643 + 1024))
    self.assertLookup(643, '<null>', obj, _Hashable(643 + 50 * 1024))
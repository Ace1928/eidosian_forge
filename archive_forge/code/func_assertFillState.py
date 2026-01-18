import sys
from breezy import tests
from breezy.tests import features
def assertFillState(self, used, fill, mask, obj):
    self.assertEqual((used, fill, mask), (obj.used, obj.fill, obj.mask))
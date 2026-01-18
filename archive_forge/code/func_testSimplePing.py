from six.moves import range
import sys
from pyu2f.tests.lib import util
def testSimplePing(self):
    dev = util.FakeHidDevice(cid_to_allocate=None)
    dev.Write([0, 0, 0, 1, 129, 0, 3, 1, 2, 3])
    self.assertEquals(dev.Read(), [0, 0, 0, 1, 129, 0, 3, 1, 2, 3] + [0 for _ in range(54)])
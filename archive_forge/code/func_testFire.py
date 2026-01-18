import sys
import fire
from fire import testutils
import mock
def testFire(self):
    with mock.patch.object(sys, 'argv', ['commandname']):
        fire.Fire()
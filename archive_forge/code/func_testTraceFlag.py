from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import fire
from fire import test_components as tc
from fire import testutils
import mock
import six
def testTraceFlag(self):
    with self.assertRaisesFireExit(0, 'Fire trace:\n'):
        fire.Fire(tc.BoolConverter, command=['as-bool', 'True', '--', '--trace'])
    with self.assertRaisesFireExit(0, 'Fire trace:\n'):
        fire.Fire(tc.BoolConverter, command=['as-bool', 'True', '--', '-t'])
    with self.assertRaisesFireExit(0, 'Fire trace:\n'):
        fire.Fire(tc.BoolConverter, command=['--', '--trace'])
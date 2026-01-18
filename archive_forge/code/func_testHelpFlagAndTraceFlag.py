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
def testHelpFlagAndTraceFlag(self):
    with self.assertRaisesFireExit(0, 'Fire trace:\n.*SYNOPSIS'):
        fire.Fire(tc.BoolConverter, command=['as-bool', 'True', '--', '--help', '--trace'])
    with self.assertRaisesFireExit(0, 'Fire trace:\n.*SYNOPSIS'):
        fire.Fire(tc.BoolConverter, command=['as-bool', 'True', '--', '-h', '-t'])
    with self.assertRaisesFireExit(0, 'Fire trace:\n.*SYNOPSIS'):
        fire.Fire(tc.BoolConverter, command=['--', '-h', '--trace'])
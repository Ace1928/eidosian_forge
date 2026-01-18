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
def testHelpFlag(self):
    with self.assertRaisesFireExit(0):
        fire.Fire(tc.BoolConverter, command=['as-bool', 'True', '--', '--help'])
    with self.assertRaisesFireExit(0):
        fire.Fire(tc.BoolConverter, command=['as-bool', 'True', '--', '-h'])
    with self.assertRaisesFireExit(0):
        fire.Fire(tc.BoolConverter, command=['--', '--help'])
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
def testFireExceptions(self):
    with self.assertRaisesFireExit(2):
        fire.Fire(tc.Empty, command=['nomethod'])
    with self.assertRaisesFireExit(2):
        fire.Fire(tc.NoDefaults, command=['double'])
    with self.assertRaisesFireExit(2):
        fire.Fire(tc.TypedProperties, command=['delta', 'x'])
    with self.assertRaises(ZeroDivisionError):
        fire.Fire(tc.NumberDefaults, command=['reciprocal', '0.0'])
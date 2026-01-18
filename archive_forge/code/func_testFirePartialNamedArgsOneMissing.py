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
def testFirePartialNamedArgsOneMissing(self):
    with self.assertRaisesFireExit(2):
        fire.Fire(tc.MixedDefaults, command=['identity'])
    with self.assertRaisesFireExit(2):
        fire.Fire(tc.MixedDefaults, command=['identity', '--beta', '2'])
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '1']), (1, '0'))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha', '1']), (1, '0'))
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
def testFireAllNamedArgs(self):
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '1', '2']), 5)
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '--alpha', '1', '2']), 5)
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '--beta', '1', '2']), 4)
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '1', '--alpha', '2']), 4)
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '1', '--beta', '2']), 5)
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '--alpha', '1', '--beta', '2']), 5)
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '--beta', '1', '--alpha', '2']), 4)
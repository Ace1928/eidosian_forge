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
def testBoolParsingLessExpectedCases(self):
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha', '10']), (10, '0'))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha', '--beta=10']), (True, 10))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', 'True', '10']), (True, 10))
    with self.assertRaisesFireExit(2):
        fire.Fire(tc.MixedDefaults, command=['identity', '--alpha', '--test'])
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha', 'True', '"--test"']), (True, '--test'))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha=--test']), ('--test', '0'))
    self.assertEqual(fire.Fire(tc.MixedDefaults, command='identity --alpha \\"--test\\"'), ('--test', '0'))
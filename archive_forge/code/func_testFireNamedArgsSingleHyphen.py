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
def testFireNamedArgsSingleHyphen(self):
    self.assertEqual(fire.Fire(tc.WithDefaults, command=['double', '-count', '5']), 10)
    self.assertEqual(fire.Fire(tc.WithDefaults, command=['triple', '-count', '5']), 15)
    self.assertEqual(fire.Fire(tc.OldStyleWithDefaults, command=['double', '-count', '5']), 10)
    self.assertEqual(fire.Fire(tc.OldStyleWithDefaults, command=['triple', '-count', '5']), 15)
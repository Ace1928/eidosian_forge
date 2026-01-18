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
def testFireObjectWithDict(self):
    self.assertEqual(fire.Fire(tc.TypedProperties, command=['delta', 'echo']), 'E')
    self.assertEqual(fire.Fire(tc.TypedProperties, command=['delta', 'echo', 'lower']), 'e')
    self.assertIsInstance(fire.Fire(tc.TypedProperties, command=['delta', 'nest']), dict)
    self.assertEqual(fire.Fire(tc.TypedProperties, command=['delta', 'nest', '0']), 'a')
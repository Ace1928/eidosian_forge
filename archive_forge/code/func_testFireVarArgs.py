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
def testFireVarArgs(self):
    self.assertEqual(fire.Fire(tc.VarArgs, command=['cumsums', 'a', 'b', 'c', 'd']), ['a', 'ab', 'abc', 'abcd'])
    self.assertEqual(fire.Fire(tc.VarArgs, command=['cumsums', '1', '2', '3', '4']), [1, 3, 6, 10])
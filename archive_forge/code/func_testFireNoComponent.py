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
def testFireNoComponent(self):
    self.assertEqual(fire.Fire(command=['tc', 'WithDefaults', 'double', '10']), 20)
    last_char = lambda text: text[-1]
    self.assertEqual(fire.Fire(command=['last_char', '"Hello"']), 'o')
    self.assertEqual(fire.Fire(command=['last-char', '"World"']), 'd')
    rset = lambda count=0: set(range(count))
    self.assertEqual(fire.Fire(command=['rset', '5']), {0, 1, 2, 3, 4})
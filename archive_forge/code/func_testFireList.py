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
def testFireList(self):
    component = ['zero', 'one', 'two', 'three']
    self.assertEqual(fire.Fire(component, command=['2']), 'two')
    self.assertEqual(fire.Fire(component, command=['3']), 'three')
    self.assertEqual(fire.Fire(component, command=['-1']), 'three')
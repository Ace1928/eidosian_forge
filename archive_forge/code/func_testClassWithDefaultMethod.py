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
def testClassWithDefaultMethod(self):
    self.assertEqual(fire.Fire(tc.DefaultMethod, command=['double', '10']), 20)
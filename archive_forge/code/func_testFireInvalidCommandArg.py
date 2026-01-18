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
def testFireInvalidCommandArg(self):
    with self.assertRaises(ValueError):
        fire.Fire(tc.WithDefaults, command=10)
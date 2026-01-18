from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import test_components as tc
from fire import testutils
from fire import trace
import mock
import six
def testOneLineResultCircularRef(self):
    circular_reference = tc.CircularReference()
    self.assertEqual(core._OneLineResult(circular_reference.create()), "{'y': {...}}")
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
def testFireUnderscoresInArg(self):
    self.assertEqual(fire.Fire(tc.Underscores, command=['underscore-function', 'example']), 'example')
    self.assertEqual(fire.Fire(tc.Underscores, command=['underscore_function', '--underscore-arg=score']), 'score')
    self.assertEqual(fire.Fire(tc.Underscores, command=['underscore_function', '--underscore_arg=score']), 'score')
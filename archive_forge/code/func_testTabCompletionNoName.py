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
def testTabCompletionNoName(self):
    completion_script = fire.Fire(tc.NoDefaults, command=['--', '--completion'])
    self.assertIn('double', completion_script)
    self.assertIn('triple', completion_script)
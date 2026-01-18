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
@testutils.skipIf(six.PY2, 'Asyncio not available in Python 2.')
def testFireAsyncio(self):
    self.assertEqual(fire.Fire(tc.py3.WithAsyncio, command=['double', '--count', '10']), 20)
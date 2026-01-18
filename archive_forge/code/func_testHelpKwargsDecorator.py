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
@testutils.skipIf(sys.version_info[0:2] <= (3, 4), 'Cannot inspect wrapped signatures in Python 2 or 3.4.')
def testHelpKwargsDecorator(self):
    with self.assertRaisesFireExit(0):
        fire.Fire(tc.decorated_method, command=['-h'])
    with self.assertRaisesFireExit(0):
        fire.Fire(tc.decorated_method, command=['--help'])
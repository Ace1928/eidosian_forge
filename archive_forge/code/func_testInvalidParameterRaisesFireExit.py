from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import test_components as tc
from fire import testutils
from fire import trace
import mock
import six
def testInvalidParameterRaisesFireExit(self):
    with self.assertRaisesFireExit(2, 'runmisspelled'):
        core.Fire(tc.Kwargs, command=['props', '--a=1', '--b=2', 'runmisspelled'])
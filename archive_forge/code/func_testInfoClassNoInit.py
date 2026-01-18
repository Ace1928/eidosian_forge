from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import unittest
from fire import inspectutils
from fire import test_components as tc
from fire import testutils
import six
def testInfoClassNoInit(self):
    info = inspectutils.Info(tc.OldStyleEmpty)
    if six.PY2:
        self.assertEqual(info.get('type_name'), 'classobj')
    else:
        self.assertEqual(info.get('type_name'), 'type')
    self.assertIn(os.path.join('fire', 'test_components.py'), info.get('file'))
    self.assertGreater(info.get('line'), 0)
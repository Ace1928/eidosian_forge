from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import unittest
from fire import inspectutils
from fire import test_components as tc
from fire import testutils
import six
@unittest.skipIf(six.PY2, 'No keyword arguments in python 2')
def testGetFullArgSpecPy3(self):
    spec = inspectutils.GetFullArgSpec(tc.py3.identity)
    self.assertEqual(spec.args, ['arg1', 'arg2', 'arg3', 'arg4'])
    self.assertEqual(spec.defaults, (10, 20))
    self.assertEqual(spec.varargs, 'arg5')
    self.assertEqual(spec.varkw, 'arg10')
    self.assertEqual(spec.kwonlyargs, ['arg6', 'arg7', 'arg8', 'arg9'])
    self.assertEqual(spec.kwonlydefaults, {'arg8': 30, 'arg9': 40})
    self.assertEqual(spec.annotations, {'arg2': int, 'arg4': int, 'arg7': int, 'arg9': int})
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import testutils
from fire import trace
def testFireTraceElementAsStringWithTargetAndLineNo(self):
    el = trace.FireTraceElement(component='Example', action='Created toy', target='Beaker', filename='beaker.py', lineno=10)
    self.assertEqual(str(el), 'Created toy "Beaker" (beaker.py:10)')
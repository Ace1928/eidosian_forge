from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import testutils
from fire import trace
def testFireTraceGetResult(self):
    t = trace.FireTrace('start')
    self.assertEqual(t.GetResult(), 'start')
    t.AddAccessedProperty('t', 'final', None, 'example.py', 10)
    self.assertEqual(t.GetResult(), 't')
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import testutils
from fire import trace
def testFireTraceHasError(self):
    t = trace.FireTrace('start')
    self.assertFalse(t.HasError())
    t.AddAccessedProperty('t', 'final', None, 'example.py', 10)
    self.assertFalse(t.HasError())
    t.AddError(ValueError('example error'), ['arg'])
    self.assertTrue(t.HasError())
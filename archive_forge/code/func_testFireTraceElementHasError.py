from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import testutils
from fire import trace
def testFireTraceElementHasError(self):
    el = trace.FireTraceElement()
    self.assertFalse(el.HasError())
    el = trace.FireTraceElement(error=ValueError('example error'))
    self.assertTrue(el.HasError())
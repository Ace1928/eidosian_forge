from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import testutils
from fire import trace
def testFireTraceElementAsStringNoMetadata(self):
    el = trace.FireTraceElement(component='Example', action='Fake action')
    self.assertEqual(str(el), 'Fake action')
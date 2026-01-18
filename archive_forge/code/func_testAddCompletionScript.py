from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import testutils
from fire import trace
def testAddCompletionScript(self):
    t = trace.FireTrace('initial object')
    t.AddCompletionScript('This is the completion script string.')
    self.assertEqual(str(t), '1. Initial component\n2. Generated completion script')
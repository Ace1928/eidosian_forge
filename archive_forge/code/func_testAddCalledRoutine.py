from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import testutils
from fire import trace
def testAddCalledRoutine(self):
    t = trace.FireTrace('initial object')
    args = ('example', 'args')
    t.AddCalledComponent('result', 'run', args, 'sample.py', 12, False, action=trace.CALLED_ROUTINE)
    self.assertEqual(str(t), '1. Initial component\n2. Called routine "run" (sample.py:12)')
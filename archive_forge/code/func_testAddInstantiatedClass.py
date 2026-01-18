from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import testutils
from fire import trace
def testAddInstantiatedClass(self):
    t = trace.FireTrace('initial object')
    args = ('example', 'args')
    t.AddCalledComponent('Classname', 'classname', args, 'sample.py', 12, False, action=trace.INSTANTIATED_CLASS)
    target = '1. Initial component\n2. Instantiated class "classname" (sample.py:12)'
    self.assertEqual(str(t), target)
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import decorators
from fire import testutils
def testSetParseFnsDefaultsFromPython(self):
    self.assertTupleEqual(WithDefaults().example1(), (10, int))
    self.assertEqual(WithDefaults().example1(5), (5, int))
    self.assertEqual(WithDefaults().example1(12.0), (12, float))
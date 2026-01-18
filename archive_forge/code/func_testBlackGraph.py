import spherogram
import unittest
from random import randrange
def testBlackGraph(self):
    repeat = 3
    while repeat > 0:
        k1 = self.random_knot()
        self.assert_(k1.black_graph().is_planar())
        repeat -= 1
    repeat = 3
    while repeat > 0:
        k2 = self.random_link()
        self.assert_(k2.black_graph().is_planar())
        repeat -= 1
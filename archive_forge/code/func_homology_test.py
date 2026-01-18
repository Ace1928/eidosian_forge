from ... import sage_helper
from .. import t3mlite as t3m
def homology_test(self):
    T = self.dual_triangulation
    B1, B2 = (self.B1(), self.B2())
    assert B1 * B2 == 0
    assert T.euler() == self.euler()
    CD = self.chain_complex()
    CT = T.chain_complex()
    assert CD.homology() == CT.homology()
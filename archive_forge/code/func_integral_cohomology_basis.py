from ... import sage_helper
from .. import t3mlite as t3m
def integral_cohomology_basis(self, dimension=1):
    assert dimension == 1
    return [OneCocycle(self, list(c.weights)) for c in self.dual_triangulation.integral_homology_basis(dimension)]
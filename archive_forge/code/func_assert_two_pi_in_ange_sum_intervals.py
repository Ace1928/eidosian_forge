from .gimbalLoopFinder import GimbalLoopFinder
from .truncatedComplex import TruncatedComplex
from .hyperbolicStructure import HyperbolicStructure
from .verificationError import *
from sage.all import matrix, prod, RealDoubleField, pi
def assert_two_pi_in_ange_sum_intervals(self):
    RIF = self.hyperbolic_structure.angle_sums[0].parent()
    two_pi = RIF(2 * pi)
    for angle in self.hyperbolic_structure.angle_sums:
        if two_pi not in angle:
            raise AngleSumIntervalNotContainingTwoPiError()
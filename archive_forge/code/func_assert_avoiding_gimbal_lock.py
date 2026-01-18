from .gimbalLoopFinder import GimbalLoopFinder
from .truncatedComplex import TruncatedComplex
from .hyperbolicStructure import HyperbolicStructure
from .verificationError import *
from sage.all import matrix, prod, RealDoubleField, pi
def assert_avoiding_gimbal_lock(self):
    d = self.gimbal_derivative()
    if not VerifyHyperbolicStructureEngine.is_invertible(d):
        raise GimbalDerivativeNotInvertibleError(d, d.determinant())
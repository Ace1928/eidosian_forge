from .computeApproxHyperbolicStructureNew import *
from .computeApproxHyperbolicStructureOrb import *
from .polishApproxHyperbolicStructure import *
from .krawczykCertifiedEdgeLengthsEngine import *
from .verifyHyperbolicStructureEngine import *
from .parseVertexGramMatrixFile import (
from snappy.snap.t3mlite import Mcomplex
def compute_verified_hyperbolic_structure_from_approx_structure(approx_hyperbolic_structure, bits_prec=53, verbose=False):
    """
    Computes a verified hyperbolic structure given an instance of
    HyperbolicStructure where the (unverified) edge lengths are in
    SageMath's RealDoubleField or RealField.
    """
    approx_hyperbolic_structure.pick_exact_and_var_edges()
    polished_hyperbolic_structure = polish_approx_hyperbolic_structure(approx_hyperbolic_structure, bits_prec, verbose=verbose)
    K = KrawczykCertifiedEdgeLengthsEngine(polished_hyperbolic_structure, bits_prec)
    result = K.partially_verified_hyperbolic_structure()
    verify_engine = VerifyHyperbolicStructureEngine(result)
    verify_engine.assert_verified_hyperbolic()
    return result
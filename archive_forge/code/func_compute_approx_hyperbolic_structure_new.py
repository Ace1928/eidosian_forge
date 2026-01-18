from .hyperbolicStructure import *
from .verificationError import *
from sage.all import RDF, pi, matrix, block_matrix, vector
def compute_approx_hyperbolic_structure_new(mcomplex, verbose=False):
    """
    Finds unverified hyperbolic structure for an Mcomplex.

    >>> from snappy import Triangulation
    >>> from snappy.snap.t3mlite import Mcomplex
    >>> isosig = 'uLLvLALLQPAPAMcbehgilknmkonpoqrqrsttxxuvcaiauxawkkutxhqqw'
    >>> m = Mcomplex(Triangulation(isosig, remove_finite_vertices = False))
    >>> h = compute_approx_hyperbolic_structure_new(m)
    >>> all([ abs(s - _two_pi) < 1e-11 for s in h.angle_sums ])
    True

    """
    global _start_edge_param
    global _iteration_stop
    edge_params = [_start_edge_param for edge in mcomplex.Edges]
    hyperbolicStructure = HyperbolicStructure(mcomplex, edge_params)
    errors_with_norm = _compute_errors_with_norm(hyperbolicStructure)
    for i in range(100):
        if verbose:
            print('Iteration: %d' % i)
        hyperbolicStructure, errors_with_norm = _adaptive_newton_step(hyperbolicStructure, errors_with_norm, verbose=verbose)
        if max([abs(x) for x in errors_with_norm[0]]) < _iteration_stop:
            return hyperbolicStructure
    raise NewtonMethodConvergenceError()
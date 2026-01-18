import string
from ..sage_helper import _within_sage, sage_method
@sage_method
def homological_longitude(manifold, cusp=None):
    """
    Returns the peripheral curve in the given cusp, if any, which is
    homologically trivial (with rational coefficients) in the manifold::

        sage: M = Manifold('m015')
        sage: M.homological_longitude()
        (2, -1)

    If no cusp is specified, the default is the first unfilled cusp;
    if all cusps are filled, the default is the first cusp::

        sage: M = Manifold('L5a1(3,4)(0,0)')
        sage: M.homological_longitude()
        (0, 1)

    The components of the next link have nontrivial linking number
    so there is no such curve::

        sage: W = Manifold('L7a2')
        sage: W.homological_longitude(cusp=1) is None
        True

    If every curve in the given cusp is trivial in the rational homology of
    the manifold, an exception is raised::

        sage: M = Manifold('4_1(1,0)')
        sage: M.homological_longitude()
        Traceback (most recent call last):
        ...
        ValueError: Every curve on cusp is homologically trivial
    """
    if cusp is None:
        unfilled = [i for i, status in enumerate(manifold.cusp_info('complete?')) if status]
        if len(unfilled):
            cusp = unfilled[0]
        else:
            cusp = 0
    G = manifold.fundamental_group()
    f = MapToFreeAbelianization(G)
    m, l = G.peripheral_curves()[cusp]
    kernel_basis = matrix(ZZ, [f(m), f(l)]).left_kernel().basis()
    if len(kernel_basis) >= 2:
        raise ValueError('Every curve on cusp is homologically trivial')
    if len(kernel_basis) == 0:
        return None
    return kernel_basis[0]
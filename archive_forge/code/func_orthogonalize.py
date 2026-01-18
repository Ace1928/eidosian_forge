from sympy.vector.coordsysrect import CoordSys3D
from sympy.vector.deloperator import Del
from sympy.vector.scalar import BaseScalar
from sympy.vector.vector import Vector, BaseVector
from sympy.vector.operators import gradient, curl, divergence
from sympy.core.function import diff
from sympy.core.singleton import S
from sympy.integrals.integrals import integrate
from sympy.simplify.simplify import simplify
from sympy.core import sympify
from sympy.vector.dyadic import Dyadic
def orthogonalize(*vlist, orthonormal=False):
    """
    Takes a sequence of independent vectors and orthogonalizes them
    using the Gram - Schmidt process. Returns a list of
    orthogonal or orthonormal vectors.

    Parameters
    ==========

    vlist : sequence of independent vectors to be made orthogonal.

    orthonormal : Optional parameter
                  Set to True if the vectors returned should be
                  orthonormal.
                  Default: False

    Examples
    ========

    >>> from sympy.vector.coordsysrect import CoordSys3D
    >>> from sympy.vector.functions import orthogonalize
    >>> C = CoordSys3D('C')
    >>> i, j, k = C.base_vectors()
    >>> v1 = i + 2*j
    >>> v2 = 2*i + 3*j
    >>> orthogonalize(v1, v2)
    [C.i + 2*C.j, 2/5*C.i + (-1/5)*C.j]

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Gram-Schmidt_process

    """
    if not all((isinstance(vec, Vector) for vec in vlist)):
        raise TypeError('Each element must be of Type Vector')
    ortho_vlist = []
    for i, term in enumerate(vlist):
        for j in range(i):
            term -= ortho_vlist[j].projection(vlist[i])
        if simplify(term).equals(Vector.zero):
            raise ValueError('Vector set not linearly independent')
        ortho_vlist.append(term)
    if orthonormal:
        ortho_vlist = [vec.normalize() for vec in ortho_vlist]
    return ortho_vlist
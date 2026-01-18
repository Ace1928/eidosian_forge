from operator import add, neg, pos, sub, mul
from collections import defaultdict
from sympy.utilities.iterables import _strongly_connected_components
from .exceptions import DMBadInputError, DMDomainError, DMShapeError
from .ddm import DDM
from .lll import ddm_lll, ddm_lll_transform
from sympy.polys.domains import QQ
@classmethod
def from_ddm(cls, ddm):
    """
        converts object of :py:class:`~.DDM` to
        :py:class:`~.SDM`

        Examples
        ========

        >>> from sympy.polys.matrices.ddm import DDM
        >>> from sympy.polys.matrices.sdm import SDM
        >>> from sympy import QQ
        >>> ddm = DDM( [[QQ(1, 2), 0], [0, QQ(3, 4)]], (2, 2), QQ)
        >>> A = SDM.from_ddm(ddm)
        >>> A
        {0: {0: 1/2}, 1: {1: 3/4}}

        """
    return cls.from_list(ddm, ddm.shape, ddm.domain)
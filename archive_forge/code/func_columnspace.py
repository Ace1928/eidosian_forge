from functools import reduce
from typing import Union as tUnion, Tuple as tTuple
from sympy.core.sympify import _sympify
from ..domains import Domain
from ..constructor import construct_domain
from .exceptions import (DMNonSquareMatrixError, DMShapeError,
from .ddm import DDM
from .sdm import SDM
from .domainscalar import DomainScalar
from sympy.polys.domains import ZZ, EXRAW, QQ
def columnspace(self):
    """
        Returns the columnspace for the DomainMatrix

        Returns
        =======

        DomainMatrix
            The columns of this matrix form a basis for the columnspace.

        Examples
        ========

        >>> from sympy import QQ
        >>> from sympy.polys.matrices import DomainMatrix
        >>> A = DomainMatrix([
        ...    [QQ(1), QQ(-1)],
        ...    [QQ(2), QQ(-2)]], (2, 2), QQ)
        >>> A.columnspace()
        DomainMatrix([[1], [2]], (2, 1), QQ)

        """
    if not self.domain.is_Field:
        raise DMNotAField('Not a field')
    rref, pivots = self.rref()
    rows, cols = self.shape
    return self.extract(range(rows), pivots)
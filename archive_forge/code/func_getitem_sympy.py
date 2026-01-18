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
def getitem_sympy(self, i, j):
    return self.domain.to_sympy(self.rep.getitem(i, j))
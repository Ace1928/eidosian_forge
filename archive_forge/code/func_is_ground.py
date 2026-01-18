from sympy.polys.domains.domain import Domain
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.polyerrors import (CoercionFailed, NotInvertible,
from sympy.polys.polytools import Poly
from sympy.printing.defaults import DefaultPrinting
@property
def is_ground(f):
    return f.rep.is_ground
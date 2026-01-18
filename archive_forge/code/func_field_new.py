from __future__ import annotations
from typing import Any
from functools import reduce
from operator import add, mul, lt, le, gt, ge
from sympy.core.expr import Expr
from sympy.core.mod import Mod
from sympy.core.numbers import Exp1
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.core.sympify import CantSympify, sympify
from sympy.functions.elementary.exponential import ExpBase
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.domains.fractionfield import FractionField
from sympy.polys.domains.polynomialring import PolynomialRing
from sympy.polys.constructor import construct_domain
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyoptions import build_options
from sympy.polys.polyutils import _parallel_dict_from_expr
from sympy.polys.rings import PolyElement
from sympy.printing.defaults import DefaultPrinting
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence
from sympy.utilities.magic import pollute
def field_new(self, element):
    if isinstance(element, FracElement):
        if self == element.field:
            return element
        if isinstance(self.domain, FractionField) and self.domain.field == element.field:
            return self.ground_new(element)
        elif isinstance(self.domain, PolynomialRing) and self.domain.ring.to_field() == element.field:
            return self.ground_new(element)
        else:
            raise NotImplementedError('conversion')
    elif isinstance(element, PolyElement):
        denom, numer = element.clear_denoms()
        if isinstance(self.domain, PolynomialRing) and numer.ring == self.domain.ring:
            numer = self.ring.ground_new(numer)
        elif isinstance(self.domain, FractionField) and numer.ring == self.domain.field.to_ring():
            numer = self.ring.ground_new(numer)
        else:
            numer = numer.set_ring(self.ring)
        denom = self.ring.ground_new(denom)
        return self.raw_new(numer, denom)
    elif isinstance(element, tuple) and len(element) == 2:
        numer, denom = list(map(self.ring.ring_new, element))
        return self.new(numer, denom)
    elif isinstance(element, str):
        raise NotImplementedError('parsing')
    elif isinstance(element, Expr):
        return self.from_expr(element)
    else:
        return self.ground_new(element)
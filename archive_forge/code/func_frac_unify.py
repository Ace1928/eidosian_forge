from sympy.core.numbers import oo
from sympy.core.sympify import CantSympify
from sympy.polys.polyerrors import CoercionFailed, NotReversible, NotInvertible
from sympy.polys.polyutils import PicklableWithSlots
from sympy.polys.densebasic import (
from sympy.polys.densearith import (
from sympy.polys.densetools import (
from sympy.polys.euclidtools import (
from sympy.polys.sqfreetools import (
from sympy.polys.factortools import (
from sympy.polys.rootisolation import (
from sympy.polys.polyerrors import (
def frac_unify(f, g):
    """Unify representations of two multivariate fractions. """
    if not isinstance(g, DMF) or f.lev != g.lev:
        raise UnificationFailed('Cannot unify %s with %s' % (f, g))
    if f.dom == g.dom and f.ring == g.ring:
        return (f.lev, f.dom, f.per, (f.num, f.den), (g.num, g.den))
    else:
        lev, dom = (f.lev, f.dom.unify(g.dom))
        ring = f.ring
        if g.ring is not None:
            if ring is not None:
                ring = ring.unify(g.ring)
            else:
                ring = g.ring
        F = (dmp_convert(f.num, lev, f.dom, dom), dmp_convert(f.den, lev, f.dom, dom))
        G = (dmp_convert(g.num, lev, g.dom, dom), dmp_convert(g.den, lev, g.dom, dom))

        def per(num, den, cancel=True, kill=False, lev=lev):
            if kill:
                if not lev:
                    return num / den
                else:
                    lev = lev - 1
            if cancel:
                num, den = dmp_cancel(num, den, lev, dom)
            return f.__class__.new((num, den), dom, lev, ring=ring)
        return (lev, dom, per, F, G)
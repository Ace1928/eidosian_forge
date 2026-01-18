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
def count_complex_roots(f, inf=None, sup=None):
    """Return the number of complex roots of ``f`` in ``[inf, sup]``. """
    return dup_count_complex_roots(f.rep, f.dom, inf=inf, sup=sup)
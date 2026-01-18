from __future__ import annotations
from typing import Any
from sympy.core.numbers import AlgebraicNumber
from sympy.core import Basic, sympify
from sympy.core.sorting import default_sort_key, ordered
from sympy.external.gmpy import HAS_GMPY
from sympy.polys.domains.domainelement import DomainElement
from sympy.polys.orderings import lex
from sympy.polys.polyerrors import UnificationFailed, CoercionFailed, DomainError
from sympy.polys.polyutils import _unify_gens, _not_a_coeff
from sympy.utilities import public
from sympy.utilities.iterables import is_sequence
def cyclotomic_field(self, n, ss=False, alias='zeta', gen=None, root_index=-1):
    """
        Convenience method to construct a cyclotomic field.

        Parameters
        ==========

        n : int
            Construct the nth cyclotomic field.
        ss : boolean, optional (default=False)
            If True, append *n* as a subscript on the alias string.
        alias : str, optional (default="zeta")
            Symbol name for the generator.
        gen : :py:class:`~.Symbol`, optional (default=None)
            Desired variable for the cyclotomic polynomial that defines the
            field. If ``None``, a dummy variable will be used.
        root_index : int, optional (default=-1)
            Specifies which root of the polynomial is desired. The ordering is
            as defined by the :py:class:`~.ComplexRootOf` class. The default of
            ``-1`` selects the root $\\mathrm{e}^{2\\pi i/n}$.

        Examples
        ========

        >>> from sympy import QQ, latex
        >>> K = QQ.cyclotomic_field(5)
        >>> K.to_sympy(K([-1, 1]))
        1 - zeta
        >>> L = QQ.cyclotomic_field(7, True)
        >>> a = L.to_sympy(L([-1, 1]))
        >>> print(a)
        1 - zeta7
        >>> print(latex(a))
        1 - \\zeta_{7}

        """
    from sympy.polys.specialpolys import cyclotomic_poly
    if ss:
        alias += str(n)
    return self.alg_field_from_poly(cyclotomic_poly(n, gen), alias=alias, root_index=root_index)
from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING
import itertools
from sympy.core import Add, Float, Mod, Mul, Number, S, Symbol, Expr
from sympy.core.alphabets import greeks
from sympy.core.containers import Tuple
from sympy.core.function import Function, AppliedUndef, Derivative
from sympy.core.operations import AssocOp
from sympy.core.power import Pow
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import SympifyError
from sympy.logic.boolalg import true, BooleanTrue, BooleanFalse
from sympy.tensor.array import NDimArray
from sympy.printing.precedence import precedence_traditional
from sympy.printing.printer import Printer, print_function
from sympy.printing.conventions import split_super_sub, requires_partial
from sympy.printing.precedence import precedence, PRECEDENCE
from mpmath.libmp.libmpf import prec_to_dps, to_str as mlib_to_str
from sympy.utilities.iterables import has_variety, sift
import re
def multiline_latex(lhs, rhs, terms_per_line=1, environment='align*', use_dots=False, **settings):
    """
    This function generates a LaTeX equation with a multiline right-hand side
    in an ``align*``, ``eqnarray`` or ``IEEEeqnarray`` environment.

    Parameters
    ==========

    lhs : Expr
        Left-hand side of equation

    rhs : Expr
        Right-hand side of equation

    terms_per_line : integer, optional
        Number of terms per line to print. Default is 1.

    environment : "string", optional
        Which LaTeX wnvironment to use for the output. Options are "align*"
        (default), "eqnarray", and "IEEEeqnarray".

    use_dots : boolean, optional
        If ``True``, ``\\\\dots`` is added to the end of each line. Default is ``False``.

    Examples
    ========

    >>> from sympy import multiline_latex, symbols, sin, cos, exp, log, I
    >>> x, y, alpha = symbols('x y alpha')
    >>> expr = sin(alpha*y) + exp(I*alpha) - cos(log(y))
    >>> print(multiline_latex(x, expr))
    \\begin{align*}
    x = & e^{i \\alpha} \\\\
    & + \\sin{\\left(\\alpha y \\right)} \\\\
    & - \\cos{\\left(\\log{\\left(y \\right)} \\right)}
    \\end{align*}

    Using at most two terms per line:
    >>> print(multiline_latex(x, expr, 2))
    \\begin{align*}
    x = & e^{i \\alpha} + \\sin{\\left(\\alpha y \\right)} \\\\
    & - \\cos{\\left(\\log{\\left(y \\right)} \\right)}
    \\end{align*}

    Using ``eqnarray`` and dots:
    >>> print(multiline_latex(x, expr, terms_per_line=2, environment="eqnarray", use_dots=True))
    \\begin{eqnarray}
    x & = & e^{i \\alpha} + \\sin{\\left(\\alpha y \\right)} \\dots\\nonumber\\\\
    & & - \\cos{\\left(\\log{\\left(y \\right)} \\right)}
    \\end{eqnarray}

    Using ``IEEEeqnarray``:
    >>> print(multiline_latex(x, expr, environment="IEEEeqnarray"))
    \\begin{IEEEeqnarray}{rCl}
    x & = & e^{i \\alpha} \\nonumber\\\\
    & & + \\sin{\\left(\\alpha y \\right)} \\nonumber\\\\
    & & - \\cos{\\left(\\log{\\left(y \\right)} \\right)}
    \\end{IEEEeqnarray}

    Notes
    =====

    All optional parameters from ``latex`` can also be used.

    """
    l = LatexPrinter(**settings)
    if environment == 'eqnarray':
        result = '\\begin{eqnarray}' + '\n'
        first_term = '& = &'
        nonumber = '\\nonumber'
        end_term = '\n\\end{eqnarray}'
        doubleet = True
    elif environment == 'IEEEeqnarray':
        result = '\\begin{IEEEeqnarray}{rCl}' + '\n'
        first_term = '& = &'
        nonumber = '\\nonumber'
        end_term = '\n\\end{IEEEeqnarray}'
        doubleet = True
    elif environment == 'align*':
        result = '\\begin{align*}' + '\n'
        first_term = '= &'
        nonumber = ''
        end_term = '\n\\end{align*}'
        doubleet = False
    else:
        raise ValueError('Unknown environment: {}'.format(environment))
    dots = ''
    if use_dots:
        dots = '\\dots'
    terms = rhs.as_ordered_terms()
    n_terms = len(terms)
    term_count = 1
    for i in range(n_terms):
        term = terms[i]
        term_start = ''
        term_end = ''
        sign = '+'
        if term_count > terms_per_line:
            if doubleet:
                term_start = '& & '
            else:
                term_start = '& '
            term_count = 1
        if term_count == terms_per_line:
            if i < n_terms - 1:
                term_end = dots + nonumber + '\\\\' + '\n'
            else:
                term_end = ''
        if term.as_ordered_factors()[0] == -1:
            term = -1 * term
            sign = '-'
        if i == 0:
            if sign == '+':
                sign = ''
            result += '{:s} {:s}{:s} {:s} {:s}'.format(l.doprint(lhs), first_term, sign, l.doprint(term), term_end)
        else:
            result += '{:s}{:s} {:s} {:s}'.format(term_start, sign, l.doprint(term), term_end)
        term_count += 1
    result += end_term
    return result
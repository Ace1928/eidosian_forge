import itertools
from functools import reduce
from sympy.core import Dummy, ilcm, Add, Mul, Pow, S
from sympy.integrals.rde import (order_at, order_at_oo, weak_normalizer,
from sympy.integrals.risch import (gcdex_diophantine, frac_in, derivation,
from sympy.polys import Poly, lcm, cancel, sqf_list
from sympy.polys.polymatrix import PolyMatrix as Matrix
from sympy.solvers import solve
def is_deriv_k(fa, fd, DE):
    """
    Checks if Df/f is the derivative of an element of k(t).

    Explanation
    ===========

    a in k(t) is the derivative of an element of k(t) if there exists b in k(t)
    such that a = Db.  Either returns (ans, u), such that Df/f == Du, or None,
    which means that Df/f is not the derivative of an element of k(t).  ans is
    a list of tuples such that Add(*[i*j for i, j in ans]) == u.  This is useful
    for seeing exactly which elements of k(t) produce u.

    This function uses the structure theorem approach, which says that for any
    f in K, Df/f is the derivative of a element of K if and only if there are ri
    in QQ such that::

            ---               ---       Dt
            \\    r  * Dt   +  \\    r  *   i      Df
            /     i     i     /     i   ---   =  --.
            ---               ---        t        f
         i in L            i in E         i
               K/C(x)            K/C(x)


    Where C = Const(K), L_K/C(x) = { i in {1, ..., n} such that t_i is
    transcendental over C(x)(t_1, ..., t_i-1) and Dt_i = Da_i/a_i, for some a_i
    in C(x)(t_1, ..., t_i-1)* } (i.e., the set of all indices of logarithmic
    monomials of K over C(x)), and E_K/C(x) = { i in {1, ..., n} such that t_i
    is transcendental over C(x)(t_1, ..., t_i-1) and Dt_i/t_i = Da_i, for some
    a_i in C(x)(t_1, ..., t_i-1) } (i.e., the set of all indices of
    hyperexponential monomials of K over C(x)).  If K is an elementary extension
    over C(x), then the cardinality of L_K/C(x) U E_K/C(x) is exactly the
    transcendence degree of K over C(x).  Furthermore, because Const_D(K) ==
    Const_D(C(x)) == C, deg(Dt_i) == 1 when t_i is in E_K/C(x) and
    deg(Dt_i) == 0 when t_i is in L_K/C(x), implying in particular that E_K/C(x)
    and L_K/C(x) are disjoint.

    The sets L_K/C(x) and E_K/C(x) must, by their nature, be computed
    recursively using this same function.  Therefore, it is required to pass
    them as indices to D (or T).  E_args are the arguments of the
    hyperexponentials indexed by E_K (i.e., if i is in E_K, then T[i] ==
    exp(E_args[i])).  This is needed to compute the final answer u such that
    Df/f == Du.

    log(f) will be the same as u up to a additive constant.  This is because
    they will both behave the same as monomials. For example, both log(x) and
    log(2*x) == log(x) + log(2) satisfy Dt == 1/x, because log(2) is constant.
    Therefore, the term const is returned.  const is such that
    log(const) + f == u.  This is calculated by dividing the arguments of one
    logarithm from the other.  Therefore, it is necessary to pass the arguments
    of the logarithmic terms in L_args.

    To handle the case where we are given Df/f, not f, use is_deriv_k_in_field().

    See also
    ========
    is_log_deriv_k_t_radical_in_field, is_log_deriv_k_t_radical

    """
    dfa, dfd = (fd * derivation(fa, DE) - fa * derivation(fd, DE), fd * fa)
    dfa, dfd = dfa.cancel(dfd, include=True)
    if len(DE.exts) != len(DE.D):
        if [i for i in DE.cases if i == 'tan'] or {i for i in DE.cases if i == 'primitive'} - set(DE.indices('log')):
            raise NotImplementedError('Real version of the structure theorems with hypertangent support is not yet implemented.')
        raise NotImplementedError('Nonelementary extensions not supported in the structure theorems.')
    E_part = [DE.D[i].quo(Poly(DE.T[i], DE.T[i])).as_expr() for i in DE.indices('exp')]
    L_part = [DE.D[i].as_expr() for i in DE.indices('log')]
    dum = Dummy()
    lhs = Matrix([E_part + L_part], dum)
    rhs = Matrix([dfa.as_expr() / dfd.as_expr()], dum)
    A, u = constant_system(lhs, rhs, DE)
    u = u.to_Matrix()
    if not A or not all((derivation(i, DE, basic=True).is_zero for i in u)):
        return None
    elif not all((i.is_Rational for i in u)):
        raise NotImplementedError('Cannot work with non-rational coefficients in this case.')
    else:
        terms = [DE.extargs[i] for i in DE.indices('exp')] + [DE.T[i] for i in DE.indices('log')]
        ans = list(zip(terms, u))
        result = Add(*[Mul(i, j) for i, j in ans])
        argterms = [DE.T[i] for i in DE.indices('exp')] + [DE.extargs[i] for i in DE.indices('log')]
        l = []
        ld = []
        for i, j in zip(argterms, u):
            i, d = i.as_numer_denom()
            icoeff, iterms = sqf_list(i)
            l.append(Mul(*[Pow(icoeff, j)] + [Pow(b, e * j) for b, e in iterms]))
            dcoeff, dterms = sqf_list(d)
            ld.append(Mul(*[Pow(dcoeff, j)] + [Pow(b, e * j) for b, e in dterms]))
        const = cancel(fa.as_expr() / fd.as_expr() / Mul(*l) * Mul(*ld))
        return (ans, result, const)
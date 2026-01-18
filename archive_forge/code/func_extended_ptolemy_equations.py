import snappy
import snappy.snap.t3mlite as t3m
import snappy.snap.peripheral as peripheral
from sage.all import ZZ, QQ, GF, gcd, PolynomialRing, cyclotomic_polynomial
def extended_ptolemy_equations(manifold, gen_obs_class=None, nonzero_cond=True, return_full_var_dict=False, notation='short'):
    """
    We assign ptolemy coordinates ['a', 'b', 'c', 'd', 'e', 'f'] to the
    *directed* edges::

        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]


    where recall that the basic orientation convention of t3m and
    SnapPy is that a positively oriented simplex is as below. ::

             1
            /|          d/ | \\e
          /  |           /   |           2----|----3   with back edge from 2 to 3 labelled f.
         \\   |   /
         b\\  |a /c
           \\ | /
            \\|/
             0

    sage: M = Manifold('m016')
    sage: I = extended_ptolemy_equations(M)
    sage: I.dimension()
    1
    """
    if gen_obs_class is None:
        gen_obs_class = manifold.ptolemy_generalized_obstruction_classes(2)[0]
    m_star, l_star = peripheral.peripheral_cohomology_basis(manifold)
    n = manifold.num_tetrahedra()
    if notation == 'short':
        var_names = ['a', 'b', 'c', 'd', 'e', 'f']
        first_var_name = 'a0'
    else:
        var_names = ['c_1100_', 'c_1010_', 'c_1001_', 'c_0110_', 'c_0101_', 'c_0011_']
        first_var_name = 'c_1100_0'
    tet_vars = [x + repr(d) for d in range(n) for x in var_names]

    def var(tet, edge):
        return tet_vars[6 * tet + directed_edges.index(edge)]
    all_arrows = arrows_around_edges(manifold)
    independent_vars = [var(a[0][0], a[0][2]) for a in all_arrows]
    assert first_var_name in independent_vars
    if nonzero_cond:
        nonzero_cond_vars = [v.swapcase() for v in independent_vars]
    else:
        nonzero_cond_vars = []
    R = PolynomialRing(QQ, ['M', 'L', 'm', 'l'] + independent_vars + nonzero_cond_vars)
    M, L, m, l = (R('M'), R('L'), R('m'), R('l'))

    def var(tet, edge):
        return tet_vars[6 * tet + directed_edges.index(edge)]
    in_terms_of_indep_vars = {v: R(v) for v in independent_vars}
    in_terms_of_indep_vars['M'] = M
    in_terms_of_indep_vars['L'] = L
    edge_gluings = EdgeGluings(gen_obs_class)
    in_terms_of_indep_vars_data = {v: (1, 0, 0, v) for v in independent_vars}
    for around_one_edge in arrows_around_edges(manifold):
        tet0, face0, edge0 = around_one_edge[0]
        indep_var = R(var(tet0, edge0))
        sign, m_e, l_e = (1, 0, 0)
        for tet1, face1, edge1 in around_one_edge[:-1]:
            (tet2, face2, edge2), a_sign = edge_gluings[tet1, face1, edge1]
            sign = a_sign * sign
            m_e -= sum((m_star[tet1, t3m.TwoSubsimplices[face1], t3m.ZeroSubsimplices[v]] for v in edge1))
            l_e -= sum((l_star[tet1, t3m.TwoSubsimplices[face1], t3m.ZeroSubsimplices[v]] for v in edge1))
            mvar = M if m_e > 0 else m
            lvar = L if l_e > 0 else l
            dep_var = var(tet2, edge2)
            in_terms_of_indep_vars_data[dep_var] = (sign, m_e, l_e, var(tet0, edge0))
            in_terms_of_indep_vars[dep_var] = sign * mvar ** abs(m_e) * lvar ** abs(l_e) * indep_var
    tet_vars = [in_terms_of_indep_vars[v] for v in tet_vars]
    rels = [R(first_var_name) - 1, M * m - 1, L * l - 1]
    for tet in range(n):
        a, b, c, d, e, f = tet_vars[6 * tet:6 * (tet + 1)]
        rels.append(simplify_equation(c * d + a * f - b * e))
    if nonzero_cond:
        for v in independent_vars:
            rels.append(R(v) * R(v.swapcase()) - 1)
    if return_full_var_dict == 'data':
        return (R.ideal(rels), in_terms_of_indep_vars_data)
    if return_full_var_dict:
        return (R.ideal(rels), in_terms_of_indep_vars)
    else:
        return R.ideal(rels)
import snappy
from sage.all import ZZ, PolynomialRing
def gluing_variety_ideal(manifold, vars_per_tet=1):
    manifold = manifold.copy()
    n = manifold.num_tetrahedra()
    var_names = ['z%d' % i for i in range(n)]
    ideal_gens = []
    if vars_per_tet != 1:
        assert vars_per_tet == 2
        var_names += ['w%d' % i for i in range(n)]
    R = PolynomialRing(ZZ, var_names)
    if vars_per_tet == 2:
        ideal_gens += [R('z%d + w%d - 1' % (i, i)) for i in range(n)]
    eqn_data = snappy.snap.shapes.enough_gluing_equations(manifold)
    ideal_gens += [make_rect_eqn(R, A, B, c, vars_per_tet) for A, B, c in eqn_data]
    return R.ideal(ideal_gens)
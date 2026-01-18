import snappy
import snappy.snap.t3mlite as t3m
import snappy.snap.peripheral as peripheral
from sage.all import ZZ, QQ, GF, gcd, PolynomialRing, cyclotomic_polynomial
def shapes_of_SL2C_reps_for_filled(manifold, phc_solver=None):
    """
    Use CyPHC to find the shapes corresponding to SL2C representations
    of the given closed manifold, as well as those which are
    boundary-parabolic with respect to the Dehn-filling description.

    sage: M = Manifold('m006(-5, 1)')
    sage: shape_sets = shapes_of_SL2C_reps_for_filled(M)
    sage: len(shape_sets)
    24
    sage: max(shapes['err'] for shapes in shape_sets) < 1e-13
    True
    """
    if phc_solver is None:
        import phc_wrapper
        phc_solver = phc_wrapper.phc_direct
    n = manifold.num_tetrahedra()
    I, var_dict = ptolemy_ideal_for_filled(manifold, nonzero_cond=False, return_full_var_dict=True)
    sols = phc_solver(I)
    vars = I.ring().gens()
    ans = []
    for sol in sols:
        indep_values = {v: sol[repr(v)] for v in vars}
        sol_dict = {v: poly.subs(indep_values) for v, poly in var_dict.items()}
        shape_dict = {'M': sol_dict['M'], 'L': sol_dict['L']}
        for i in range(n):
            i = repr(i)
            top = sol_dict['b' + i] * sol_dict['e' + i]
            bottom = sol_dict['c' + i] * sol_dict['d' + i]
            shape_dict['z' + i] = clean_complex(top / bottom)
        for attr in ['err', 'rco', 'res', 'mult']:
            shape_dict[attr] = sol[attr]
        ans.append(shape_dict)
    return ans
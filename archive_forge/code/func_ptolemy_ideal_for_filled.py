import snappy
import snappy.snap.t3mlite as t3m
import snappy.snap.peripheral as peripheral
from sage.all import ZZ, QQ, GF, gcd, PolynomialRing, cyclotomic_polynomial
def ptolemy_ideal_for_filled(manifold, nonzero_cond=True, return_full_var_dict=False, notation='short'):
    assert manifold.cusp_info('is_complete') == [False]
    a, b = [int(x) for x in manifold.cusp_info(0)['filling']]
    I, var_dict = extended_ptolemy_equations(manifold, nonzero_cond=nonzero_cond, return_full_var_dict=True if not return_full_var_dict else return_full_var_dict, notation=notation)
    R = I.ring()
    if (a, b) == (1, 0):
        new_gens = [p.subs(M=1, m=1) for p in I.gens()] + [R('M - 1'), R('m - 1')]
        I = R.ideal([p for p in new_gens if p != 0])
    else:
        mvar = R('M') if a > 0 else R('m')
        lvar = R('l') if b > 0 else R('L')
        I = I + [mvar ** abs(a) - lvar ** abs(b)]
    if return_full_var_dict:
        return (I, var_dict)
    else:
        return I
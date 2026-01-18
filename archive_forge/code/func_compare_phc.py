import sys, snappy, giac_rur, extended, phc_wrapper, time, gluing
from sage.all import QQ, PolynomialRing, CC, QQbar, macaulay2
def compare_phc(manifold):
    start = time.time()
    sols1 = hash_sols(ptolemy_phc_as_used(manifold))
    print('Ptolemy (as used): %d solutions in %.2f' % (len(sols1), time.time() - start))
    start = time.time()
    alt_manifold = manifold.copy()
    alt_manifold.set_peripheral_curves('fillings')
    sols4 = hash_sols(ptolemy_phc_as_used(alt_manifold))
    print('Ptolemy (as used, with merid changed): %d solutions in %.2f' % (len(sols4), time.time() - start))
    print('Overlap 1 and 4: %d' % len(sols1.intersection(sols4)))
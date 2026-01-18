import snappy
import snappy.snap.t3mlite as t3m
import snappy.snap.peripheral as peripheral
from sage.all import ZZ, QQ, GF, gcd, PolynomialRing, cyclotomic_polynomial
def parse_ptolemy_face(var):
    s, index, tet = var.split('_')
    return (int(tet), int(index))
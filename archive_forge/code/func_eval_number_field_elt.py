import operator
from functools import reduce
from ..math_basics import prod
from ..sage_helper import _within_sage, sage_method, SageNotAvailable
from .realAlgebra import field_containing_real_and_imaginary_part_of_number_field
def eval_number_field_elt(elt, root):
    if elt.is_zero():
        return _Zero
    poly = elt.lift()
    R = poly.base_ring()
    coeffs = poly.coefficients()
    exps = poly.exponents()
    powers = [R(1)]
    for i in range(max(exps)):
        powers.append(powers[-1] * root)
    return sum((c * powers[e] for c, e in zip(coeffs, exps)))
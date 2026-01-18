import operator
from functools import reduce
from ..math_basics import prod
from ..sage_helper import _within_sage, sage_method, SageNotAvailable
from .realAlgebra import field_containing_real_and_imaginary_part_of_number_field
def sign_with_interval(self):
    """
        Similar to sign, but for the non-zero case, also return the interval
        certifying the sign - useful for debugging.
        """
    prec = 106
    numerical_sign, interval_val = self._sign_numerical(prec)
    if numerical_sign is not None:
        return (numerical_sign, interval_val)
    if self == 0:
        return (0, 0)
    while True:
        prec *= 2
        numerical_sign, interval_val = self._sign_numerical(prec)
        if numerical_sign is not None:
            return (numerical_sign, interval_val)
from ...sage_helper import _within_sage, sage_method
from .extended_bloch import *
from ...snap import t3mlite as t3m
@sage_method
def normalize_by_pi_square_over_two(z):
    """
    Add multiples of pi^2/2 to the real part to try to bring the
    real part between -pi^2/4 and pi^2/4.
    """
    CIF = z.parent()
    RIF = CIF.real_field()
    pi_square_over_two = RIF(pi ** 2 / 2)
    q = (z.real().center() / pi_square_over_two.center()).round()
    return z - q * pi_square_over_two
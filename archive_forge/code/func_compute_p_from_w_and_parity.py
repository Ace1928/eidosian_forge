from ...sage_helper import _within_sage, sage_method
@sage_method
def compute_p_from_w_and_parity(w, parity):
    """
    Compute p such that w - p * pi * i should have imaginary part between
    -pi and pi and p has the same parity as the given value for parity
    (the given value is supposed to be 0 or 1).

    Note that this computation is not verified.
    """
    RF = RealField(w.parent().precision())
    real_part = (w.imag().center() / RF(pi) - parity) / 2
    return 2 * Integer(real_part.round()) + parity
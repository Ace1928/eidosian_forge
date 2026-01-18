from ...sage_helper import _within_sage, sage_method
@sage_method
def compute_z_p_q_from_flattening_w0_w1(w0, w1):
    """
    Given w0 and w1 such that +- exp(w0) +- exp(-w1) = 1, compute
    a triple [z; p, q] such that
    w0 = log(z) + p * pi * i and w1 = -log(1-z) + q * pi * i.

    While z is and the parities of p and q are verified, p and q are
    not verified in the following sense:
    w0 - p * pi * i and w1 + q * pi * i are likely to have imaginary
    part between -pi and pi, but this is not verified.
    """
    z, p_parity, q_parity = compute_z_and_parities_from_flattening_w0_w1(w0, w1)
    return (z, compute_p_from_w_and_parity(w0, p_parity), compute_p_from_w_and_parity(w1, q_parity))
from ...sage_helper import _within_sage, sage_method
@sage_method
def compute_Neumanns_Rogers_dilog_from_flattening_w0_w1(w0, w1):
    """
    Given a flattening w0, w1 such that +- exp(w0) +- exp(-w1) = 1, compute
    the complex volume given by R(z;p,q) (equation before Proposition 2.5 in
    Neumann's Extended Bloch group and the Cheeger-Chern-Simons class).
    """
    RIF = w0.parent().real_field()
    my_pi = RIF(pi)
    z, p, q = compute_z_p_q_from_flattening_w0_w1(w0, w1)
    logZ = w0 - my_pi * p * sage.all.I
    logOneMinusZ = -(w1 - my_pi * q * sage.all.I)
    term1 = logZ * logOneMinusZ
    term2 = my_pi * sage.all.I * (p * logOneMinusZ + q * logZ)
    if z.real().center() < 0.5:
        if not is_imaginary_part_bounded(logOneMinusZ, 2):
            raise Exception('Problem with computing Neumanns dilog using (1)', z, logOneMinusZ)
        return (term1 + term2) / 2 + my_dilog(z) - my_pi * my_pi / 6
    else:
        if not is_imaginary_part_bounded(logZ, 2):
            raise Exception('Problem with computing Neumanns dilog using (2)', z, logZ)
        return (-term1 + term2) / 2 - my_dilog(1 - z)
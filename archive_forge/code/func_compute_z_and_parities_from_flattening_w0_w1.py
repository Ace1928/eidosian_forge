from ...sage_helper import _within_sage, sage_method
@sage_method
def compute_z_and_parities_from_flattening_w0_w1(w0, w1):
    """
    Given a pair (w0, w1) with +- exp(w0) +- exp(-w1) = 1, compute (z, p, q)
    such that z = (-1)^p * exp(w0) and 1/(1-z) = (-1)^q exp(w1)
    where p, q in {0,1}.
    """
    e0 = exp(w0)
    e1 = exp(-w1)
    l = [((-1) ** p * e0, p, q) for p in [0, 1] for q in [0, 1] if Integer(1) in (-1) ** p * e0 + (-1) ** q * e1]
    if not len(l) == 1:
        raise Exception('Bad flattening %s %s %s' % (w0, w1, len(l)))
    return l[0]
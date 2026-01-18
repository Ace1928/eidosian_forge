from .chemistry import equilibrium_quotient
def equilibrium_residual(rc, c0, stoich, K, activity_product=None):
    """
    Parameters
    ---------
    rc: float
        Reaction coordinate
    c0: array_like of reals
        concentrations
    stoich: tuple
        per specie stoichiometry coefficient
    K: float
        equilibrium constant
    activity_product: callable
        callback for calculating the activity product taking
        concentration as single parameter.
    """
    if not hasattr(stoich, 'ndim') or stoich.ndim == 1:
        c = c0 + stoich * rc
    else:
        c = c0 + np.dot(stoich, rc)
    Q = equilibrium_quotient(c, stoich)
    if activity_product is not None:
        Q *= activity_product(c)
    return K - Q
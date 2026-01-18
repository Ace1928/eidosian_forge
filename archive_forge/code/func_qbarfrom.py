from .functions import defun, defun_wrapped
@defun_wrapped
def qbarfrom(ctx, q=None, m=None, k=None, tau=None, qbar=None):
    """
    Returns the number-theoretic nome `\\bar q`, given any of
    `q, m, k, \\tau, \\bar{q}`::

        >>> from mpmath import *
        >>> mp.dps = 25; mp.pretty = True
        >>> qbarfrom(qbar=0.25)
        0.25
        >>> qbarfrom(q=qfrom(qbar=0.25))
        0.25
        >>> qbarfrom(m=extraprec(20)(mfrom)(qbar=0.25))  # ill-conditioned
        0.25
        >>> qbarfrom(k=extraprec(20)(kfrom)(qbar=0.25))  # ill-conditioned
        0.25
        >>> qbarfrom(tau=taufrom(qbar=0.25))
        (0.25 + 0.0j)

    """
    if qbar is not None:
        return ctx.convert(qbar)
    if q is not None:
        return ctx.convert(q) ** 2
    if m is not None:
        return nome(ctx, m) ** 2
    if k is not None:
        return nome(ctx, ctx.convert(k) ** 2) ** 2
    if tau is not None:
        return ctx.expjpi(2 * tau)
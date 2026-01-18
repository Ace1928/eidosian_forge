import cupy
def residuez(b, a, tol=0.001, rtype='avg'):
    """Compute partial-fraction expansion of b(z) / a(z).

    If `M` is the degree of numerator `b` and `N` the degree of denominator
    `a`::

                b(z)     b[0] + b[1] z**(-1) + ... + b[M] z**(-M)
        H(z) = ------ = ------------------------------------------
                a(z)     a[0] + a[1] z**(-1) + ... + a[N] z**(-N)

    then the partial-fraction expansion H(z) is defined as::

                 r[0]                   r[-1]
         = --------------- + ... + ---------------- + k[0] + k[1]z**(-1) ...
           (1-p[0]z**(-1))         (1-p[-1]z**(-1))

    If there are any repeated roots (closer than `tol`), then the partial
    fraction expansion has terms like::

             r[i]              r[i+1]                    r[i+n-1]
        -------------- + ------------------ + ... + ------------------
        (1-p[i]z**(-1))  (1-p[i]z**(-1))**2         (1-p[i]z**(-1))**n

    This function is used for polynomials in negative powers of z,
    such as digital filters in DSP.  For positive powers, use `residue`.

    See Notes of `residue` for details about the algorithm.

    Parameters
    ----------
    b : array_like
        Numerator polynomial coefficients.
    a : array_like
        Denominator polynomial coefficients.
    tol : float, optional
        The tolerance for two roots to be considered equal in terms of
        the distance between them. Default is 1e-3. See `unique_roots`
        for further details.
    rtype : {'avg', 'min', 'max'}, optional
        Method for computing a root to represent a group of identical roots.
        Default is 'avg'. See `unique_roots` for further details.

    Returns
    -------
    r : ndarray
        Residues corresponding to the poles. For repeated poles, the residues
        are ordered to correspond to ascending by power fractions.
    p : ndarray
        Poles ordered by magnitude in ascending order.
    k : ndarray
        Coefficients of the direct polynomial term.

    Warning
    -------
    This function may synchronize the device.

    See Also
    --------
    scipy.signal.residuez
    invresz, residue, unique_roots
    """
    if cupy.issubdtype(b.dtype, cupy.complexfloating) or cupy.issubdtype(a.dtype, cupy.complexfloating):
        b = b.astype(complex)
        a = a.astype(complex)
    else:
        b = b.astype(float)
        a = a.astype(float)
    b = cupy.trim_zeros(cupy.atleast_1d(b), 'b')
    a = cupy.trim_zeros(cupy.atleast_1d(a), 'b')
    if a.size == 0:
        raise ValueError('Denominator `a` is zero.')
    elif a[0] == 0:
        raise ValueError('First coefficient of determinant `a` must be non-zero.')
    poles = roots(a)
    if b.size == 0:
        return (cupy.zeros(poles.shape), _cmplx_sort(poles)[0], cupy.array([]))
    b_rev = b[::-1]
    a_rev = a[::-1]
    if len(b_rev) < len(a_rev):
        k_rev = cupy.empty(0)
    else:
        k_rev, b_rev = _polydiv(b_rev, a_rev)
    unique_poles, multiplicity = unique_roots(poles, tol=tol, rtype=rtype)
    unique_poles, order = _cmplx_sort(unique_poles)
    multiplicity = multiplicity[order]
    residues = _compute_residues(1 / unique_poles, multiplicity, b_rev)
    index = 0
    powers = cupy.empty(len(residues), dtype=int)
    for pole, mult in zip(unique_poles, multiplicity):
        poles[index:index + mult] = pole
        powers[index:index + mult] = 1 + cupy.arange(int(mult))
        index += mult
    residues *= (-poles) ** powers / a_rev[0]
    return (residues, poles, k_rev[::-1])
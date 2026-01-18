import cupy
def residue(b, a, tol=0.001, rtype='avg'):
    """Compute partial-fraction expansion of b(s) / a(s).

    If `M` is the degree of numerator `b` and `N` the degree of denominator
    `a`::

              b(s)     b[0] s**(M) + b[1] s**(M-1) + ... + b[M]
      H(s) = ------ = ------------------------------------------
              a(s)     a[0] s**(N) + a[1] s**(N-1) + ... + a[N]

    then the partial-fraction expansion H(s) is defined as::

               r[0]       r[1]             r[-1]
           = -------- + -------- + ... + --------- + k(s)
             (s-p[0])   (s-p[1])         (s-p[-1])

    If there are any repeated roots (closer together than `tol`), then H(s)
    has terms like::

          r[i]      r[i+1]              r[i+n-1]
        -------- + ----------- + ... + -----------
        (s-p[i])  (s-p[i])**2          (s-p[i])**n

    This function is used for polynomials in positive powers of s or z,
    such as analog filters or digital filters in controls engineering.  For
    negative powers of z (typical for digital filters in DSP), use `residuez`.

    See Notes for details about the algorithm.

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
    scipy.signal.residue
    invres, residuez, numpy.poly, unique_roots

    Notes
    -----
    The "deflation through subtraction" algorithm is used for
    computations --- method 6 in [1]_.

    The form of partial fraction expansion depends on poles multiplicity in
    the exact mathematical sense. However there is no way to exactly
    determine multiplicity of roots of a polynomial in numerical computing.
    Thus you should think of the result of `residue` with given `tol` as
    partial fraction expansion computed for the denominator composed of the
    computed poles with empirically determined multiplicity. The choice of
    `tol` can drastically change the result if there are close poles.

    References
    ----------
    .. [1] J. F. Mahoney, B. D. Sivazlian, "Partial fractions expansion: a
           review of computational methodology and efficiency", Journal of
           Computational and Applied Mathematics, Vol. 9, 1983.
    """
    if cupy.issubdtype(b.dtype, cupy.complexfloating) or cupy.issubdtype(a.dtype, cupy.complexfloating):
        b = b.astype(complex)
        a = a.astype(complex)
    else:
        b = b.astype(float)
        a = a.astype(float)
    b = cupy.trim_zeros(cupy.atleast_1d(b), 'f')
    a = cupy.trim_zeros(cupy.atleast_1d(a), 'f')
    if a.size == 0:
        raise ValueError('Denominator `a` is zero.')
    poles = roots(a)
    if b.size == 0:
        return (cupy.zeros(poles.shape), _cmplx_sort(poles)[0], cupy.array([]))
    if len(b) < len(a):
        k = cupy.empty(0)
    else:
        k, b = _polydiv(b, a)
    unique_poles, multiplicity = unique_roots(poles, tol=tol, rtype=rtype)
    unique_poles, order = _cmplx_sort(unique_poles)
    multiplicity = multiplicity[order]
    residues = _compute_residues(unique_poles, multiplicity, b)
    index = 0
    for pole, mult in zip(unique_poles, multiplicity):
        poles[index:index + mult] = pole
        index += mult
    return (residues / a[0], poles, k)
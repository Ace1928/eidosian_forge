from ..libmp.backend import xrange
@defun
def eig_sort(ctx, E, EL=False, ER=False, f='real'):
    """
    This routine sorts the eigenvalues and eigenvectors delivered by ``eig``.

    parameters:
      E  : the eigenvalues as delivered by eig
      EL : the left  eigenvectors as delivered by eig, or false
      ER : the right eigenvectors as delivered by eig, or false
      f  : either a string ("real" sort by increasing real part, "imag" sort by
           increasing imag part, "abs" sort by absolute value) or a function
           mapping complexs to the reals, i.e. ``f = lambda x: -mp.re(x) ``
           would sort the eigenvalues by decreasing real part.

    return values:
       E            if EL and ER are both false.
      (E, ER)       if ER is not false and left is false.
      (E, EL)       if EL is not false and right is false.
      (E, EL, ER)   if EL and ER are not false.

    example:
      >>> from mpmath import mp
      >>> A = mp.matrix([[3, -1, 2], [2, 5, -5], [-2, -3, 7]])
      >>> E, EL, ER = mp.eig(A,left = True, right = True)
      >>> E, EL, ER = mp.eig_sort(E, EL, ER)
      >>> mp.nprint(E)
      [2.0, 4.0, 9.0]
      >>> E, EL, ER = mp.eig_sort(E, EL, ER,f = lambda x: -mp.re(x))
      >>> mp.nprint(E)
      [9.0, 4.0, 2.0]
      >>> print(mp.chop(A * ER[:,0] - E[0] * ER[:,0]))
      [0.0]
      [0.0]
      [0.0]
      >>> print(mp.chop( EL[0,:] * A - EL[0,:] * E[0]))
      [0.0  0.0  0.0]
    """
    if isinstance(f, str):
        if f == 'real':
            f = ctx.re
        elif f == 'imag':
            f = ctx.im
        elif f == 'abs':
            f = abs
        else:
            raise RuntimeError('unknown function %s' % f)
    n = len(E)
    for i in xrange(n):
        imax = i
        s = f(E[i])
        for j in xrange(i + 1, n):
            c = f(E[j])
            if c < s:
                s = c
                imax = j
        if imax != i:
            z = E[i]
            E[i] = E[imax]
            E[imax] = z
            if not isinstance(EL, bool):
                for j in xrange(n):
                    z = EL[i, j]
                    EL[i, j] = EL[imax, j]
                    EL[imax, j] = z
            if not isinstance(ER, bool):
                for j in xrange(n):
                    z = ER[j, i]
                    ER[j, i] = ER[j, imax]
                    ER[j, imax] = z
    if isinstance(EL, bool) and isinstance(ER, bool):
        return E
    if isinstance(EL, bool) and (not isinstance(ER, bool)):
        return (E, ER)
    if isinstance(ER, bool) and (not isinstance(EL, bool)):
        return (E, EL)
    return (E, EL, ER)
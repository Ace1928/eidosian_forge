import numpy
from numpy import asarray_chkfinite, single, asarray, array
from numpy.linalg import norm
from ._misc import LinAlgError, _datacopied
from .lapack import get_lapack_funcs
from ._decomp import eigvals
def schur(a, output='real', lwork=None, overwrite_a=False, sort=None, check_finite=True):
    """
    Compute Schur decomposition of a matrix.

    The Schur decomposition is::

        A = Z T Z^H

    where Z is unitary and T is either upper-triangular, or for real
    Schur decomposition (output='real'), quasi-upper triangular. In
    the quasi-triangular form, 2x2 blocks describing complex-valued
    eigenvalue pairs may extrude from the diagonal.

    Parameters
    ----------
    a : (M, M) array_like
        Matrix to decompose
    output : {'real', 'complex'}, optional
        Construct the real or complex Schur decomposition (for real matrices).
    lwork : int, optional
        Work array size. If None or -1, it is automatically computed.
    overwrite_a : bool, optional
        Whether to overwrite data in a (may improve performance).
    sort : {None, callable, 'lhp', 'rhp', 'iuc', 'ouc'}, optional
        Specifies whether the upper eigenvalues should be sorted. A callable
        may be passed that, given a eigenvalue, returns a boolean denoting
        whether the eigenvalue should be sorted to the top-left (True).
        Alternatively, string parameters may be used::

            'lhp'   Left-hand plane (x.real < 0.0)
            'rhp'   Right-hand plane (x.real > 0.0)
            'iuc'   Inside the unit circle (x*x.conjugate() <= 1.0)
            'ouc'   Outside the unit circle (x*x.conjugate() > 1.0)

        Defaults to None (no sorting).
    check_finite : bool, optional
        Whether to check that the input matrix contains only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination) if the inputs do contain infinities or NaNs.

    Returns
    -------
    T : (M, M) ndarray
        Schur form of A. It is real-valued for the real Schur decomposition.
    Z : (M, M) ndarray
        An unitary Schur transformation matrix for A.
        It is real-valued for the real Schur decomposition.
    sdim : int
        If and only if sorting was requested, a third return value will
        contain the number of eigenvalues satisfying the sort condition.

    Raises
    ------
    LinAlgError
        Error raised under three conditions:

        1. The algorithm failed due to a failure of the QR algorithm to
           compute all eigenvalues.
        2. If eigenvalue sorting was requested, the eigenvalues could not be
           reordered due to a failure to separate eigenvalues, usually because
           of poor conditioning.
        3. If eigenvalue sorting was requested, roundoff errors caused the
           leading eigenvalues to no longer satisfy the sorting condition.

    See Also
    --------
    rsf2csf : Convert real Schur form to complex Schur form

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.linalg import schur, eigvals
    >>> A = np.array([[0, 2, 2], [0, 1, 2], [1, 0, 1]])
    >>> T, Z = schur(A)
    >>> T
    array([[ 2.65896708,  1.42440458, -1.92933439],
           [ 0.        , -0.32948354, -0.49063704],
           [ 0.        ,  1.31178921, -0.32948354]])
    >>> Z
    array([[0.72711591, -0.60156188, 0.33079564],
           [0.52839428, 0.79801892, 0.28976765],
           [0.43829436, 0.03590414, -0.89811411]])

    >>> T2, Z2 = schur(A, output='complex')
    >>> T2
    array([[ 2.65896708, -1.22839825+1.32378589j,  0.42590089+1.51937378j], # may vary
           [ 0.        , -0.32948354+0.80225456j, -0.59877807+0.56192146j],
           [ 0.        ,  0.                    , -0.32948354-0.80225456j]])
    >>> eigvals(T2)
    array([2.65896708, -0.32948354+0.80225456j, -0.32948354-0.80225456j])

    An arbitrary custom eig-sorting condition, having positive imaginary part,
    which is satisfied by only one eigenvalue

    >>> T3, Z3, sdim = schur(A, output='complex', sort=lambda x: x.imag > 0)
    >>> sdim
    1

    """
    if output not in ['real', 'complex', 'r', 'c']:
        raise ValueError("argument must be 'real', or 'complex'")
    if check_finite:
        a1 = asarray_chkfinite(a)
    else:
        a1 = asarray(a)
    if len(a1.shape) != 2 or a1.shape[0] != a1.shape[1]:
        raise ValueError('expected square matrix')
    typ = a1.dtype.char
    if output in ['complex', 'c'] and typ not in ['F', 'D']:
        if typ in _double_precision:
            a1 = a1.astype('D')
            typ = 'D'
        else:
            a1 = a1.astype('F')
            typ = 'F'
    overwrite_a = overwrite_a or _datacopied(a1, a)
    gees, = get_lapack_funcs(('gees',), (a1,))
    if lwork is None or lwork == -1:
        result = gees(lambda x: None, a1, lwork=-1)
        lwork = result[-2][0].real.astype(numpy.int_)
    if sort is None:
        sort_t = 0

        def sfunction(x):
            return None
    else:
        sort_t = 1
        if callable(sort):
            sfunction = sort
        elif sort == 'lhp':

            def sfunction(x):
                return x.real < 0.0
        elif sort == 'rhp':

            def sfunction(x):
                return x.real >= 0.0
        elif sort == 'iuc':

            def sfunction(x):
                return abs(x) <= 1.0
        elif sort == 'ouc':

            def sfunction(x):
                return abs(x) > 1.0
        else:
            raise ValueError("'sort' parameter must either be 'None', or a callable, or one of ('lhp','rhp','iuc','ouc')")
    result = gees(sfunction, a1, lwork=lwork, overwrite_a=overwrite_a, sort_t=sort_t)
    info = result[-1]
    if info < 0:
        raise ValueError(f'illegal value in {-info}-th argument of internal gees')
    elif info == a1.shape[0] + 1:
        raise LinAlgError('Eigenvalues could not be separated for reordering.')
    elif info == a1.shape[0] + 2:
        raise LinAlgError('Leading eigenvalues do not satisfy sort condition.')
    elif info > 0:
        raise LinAlgError('Schur form not found. Possibly ill-conditioned.')
    if sort_t == 0:
        return (result[0], result[-3])
    else:
        return (result[0], result[-3], result[1])
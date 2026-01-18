from numpy import asanyarray, asarray, array, zeros
from scipy.sparse.linalg._interface import aslinearoperator, LinearOperator, \
def make_system(A, M, x0, b):
    """Make a linear system Ax=b

    Parameters
    ----------
    A : LinearOperator
        sparse or dense matrix (or any valid input to aslinearoperator)
    M : {LinearOperator, Nones}
        preconditioner
        sparse or dense matrix (or any valid input to aslinearoperator)
    x0 : {array_like, str, None}
        initial guess to iterative method.
        ``x0 = 'Mb'`` means using the nonzero initial guess ``M @ b``.
        Default is `None`, which means using the zero initial guess.
    b : array_like
        right hand side

    Returns
    -------
    (A, M, x, b, postprocess)
        A : LinearOperator
            matrix of the linear system
        M : LinearOperator
            preconditioner
        x : rank 1 ndarray
            initial guess
        b : rank 1 ndarray
            right hand side
        postprocess : function
            converts the solution vector to the appropriate
            type and dimensions (e.g. (N,1) matrix)

    """
    A_ = A
    A = aslinearoperator(A)
    if A.shape[0] != A.shape[1]:
        raise ValueError(f'expected square matrix, but got shape={(A.shape,)}')
    N = A.shape[0]
    b = asanyarray(b)
    if not (b.shape == (N, 1) or b.shape == (N,)):
        raise ValueError(f'shapes of A {A.shape} and b {b.shape} are incompatible')
    if b.dtype.char not in 'fdFD':
        b = b.astype('d')

    def postprocess(x):
        return x
    if hasattr(A, 'dtype'):
        xtype = A.dtype.char
    else:
        xtype = A.matvec(b).dtype.char
    xtype = coerce(xtype, b.dtype.char)
    b = asarray(b, dtype=xtype)
    b = b.ravel()
    if M is None:
        if hasattr(A_, 'psolve'):
            psolve = A_.psolve
        else:
            psolve = id
        if hasattr(A_, 'rpsolve'):
            rpsolve = A_.rpsolve
        else:
            rpsolve = id
        if psolve is id and rpsolve is id:
            M = IdentityOperator(shape=A.shape, dtype=A.dtype)
        else:
            M = LinearOperator(A.shape, matvec=psolve, rmatvec=rpsolve, dtype=A.dtype)
    else:
        M = aslinearoperator(M)
        if A.shape != M.shape:
            raise ValueError('matrix and preconditioner have different shapes')
    if x0 is None:
        x = zeros(N, dtype=xtype)
    elif isinstance(x0, str):
        if x0 == 'Mb':
            bCopy = b.copy()
            x = M.matvec(bCopy)
    else:
        x = array(x0, dtype=xtype)
        if not (x.shape == (N, 1) or x.shape == (N,)):
            raise ValueError(f'shapes of A {A.shape} and x0 {x.shape} are incompatible')
        x = x.ravel()
    return (A, M, x, b, postprocess)
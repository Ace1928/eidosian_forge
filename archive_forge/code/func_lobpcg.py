import warnings
import numpy
import cupy
import cupy.linalg as linalg
from cupyx.scipy.sparse import linalg as splinalg
def lobpcg(A, X, B=None, M=None, Y=None, tol=None, maxiter=None, largest=True, verbosityLevel=0, retLambdaHistory=False, retResidualNormsHistory=False):
    """Locally Optimal Block Preconditioned Conjugate Gradient Method (LOBPCG)

    LOBPCG is a preconditioned eigensolver for large symmetric positive
    definite (SPD) generalized eigenproblems.

    Args:
        A (array-like): The symmetric linear operator of the problem,
            usually a sparse matrix. Can be of the following types
            - cupy.ndarray
            - cupyx.scipy.sparse.csr_matrix
            - cupy.scipy.sparse.linalg.LinearOperator
        X (cupy.ndarray): Initial approximation to the ``k``
            eigenvectors (non-sparse). If `A` has ``shape=(n,n)``
            then `X` should have shape ``shape=(n,k)``.
        B (array-like): The right hand side operator in a generalized
            eigenproblem. By default, ``B = Identity``.
            Can be of following types:
            - cupy.ndarray
            - cupyx.scipy.sparse.csr_matrix
            - cupy.scipy.sparse.linalg.LinearOperator
        M (array-like): Preconditioner to `A`; by default ``M = Identity``.
            `M` should approximate the inverse of `A`.
            Can be of the following types:
            - cupy.ndarray
            - cupyx.scipy.sparse.csr_matrix
            - cupy.scipy.sparse.linalg.LinearOperator
        Y (cupy.ndarray):
            `n-by-sizeY` matrix of constraints (non-sparse), `sizeY < n`
            The iterations will be performed in the B-orthogonal complement
            of the column-space of Y. Y must be full rank.
        tol (float):
            Solver tolerance (stopping criterion).
            The default is ``tol=n*sqrt(eps)``.
        maxiter (int):
            Maximum number of iterations.  The default is ``maxiter = 20``.
        largest (bool):
            When True, solve for the largest eigenvalues,
            otherwise the smallest.
        verbosityLevel (int):
            Controls solver output.  The default is ``verbosityLevel=0``.
        retLambdaHistory (bool):
            Whether to return eigenvalue history.  Default is False.
        retResidualNormsHistory (bool):
            Whether to return history of residual norms.  Default is False.

    Returns:
        tuple:
            - `w` (cupy.ndarray): Array of ``k`` eigenvalues
            - `v` (cupy.ndarray) An array of ``k`` eigenvectors.
              `v` has the same shape as `X`.
            - `lambdas` (list of cupy.ndarray): The eigenvalue history,
              if `retLambdaHistory` is True.
            - `rnorms` (list of cupy.ndarray): The history of residual norms,
              if `retResidualNormsHistory` is True.

    .. seealso:: :func:`scipy.sparse.linalg.lobpcg`

    .. note::
        If both ``retLambdaHistory`` and ``retResidualNormsHistory`` are `True`
        the return tuple has the following format
        ``(lambda, V, lambda history, residual norms history)``.
    """
    blockVectorX = X
    blockVectorY = Y
    residualTolerance = tol
    if maxiter is None:
        maxiter = 20
    if blockVectorY is not None:
        sizeY = blockVectorY.shape[1]
    else:
        sizeY = 0
    if len(blockVectorX.shape) != 2:
        raise ValueError('expected rank-2 array for argument X')
    n, sizeX = blockVectorX.shape
    if verbosityLevel:
        aux = 'Solving '
        if B is None:
            aux += 'standard'
        else:
            aux += 'generalized'
        aux += ' eigenvalue problem with'
        if M is None:
            aux += 'out'
        aux += ' preconditioning\n\n'
        aux += 'matrix size %d\n' % n
        aux += 'block size %d\n\n' % sizeX
        if blockVectorY is None:
            aux += 'No constraints\n\n'
        elif sizeY > 1:
            aux += '%d constraints\n\n' % sizeY
        else:
            aux += '%d constraint\n\n' % sizeY
        print(aux)
    A = _makeOperator(A, (n, n))
    B = _makeOperator(B, (n, n))
    M = _makeOperator(M, (n, n))
    if n - sizeY < 5 * sizeX:
        sizeX = min(sizeX, n)
        if blockVectorY is not None:
            raise NotImplementedError('The dense eigensolver does not support constraints.')
        A_dense = A(cupy.eye(n, dtype=A.dtype))
        B_dense = None if B is None else B(cupy.eye(n, dtype=B.dtype))
        vals, vecs = _eigh(A_dense, B_dense)
        if largest:
            vals = vals[::-1]
            vecs = vecs[:, ::-1]
        vals = vals[:sizeX]
        vecs = vecs[:, :sizeX]
        return (vals, vecs)
    if residualTolerance is None or residualTolerance <= 0.0:
        residualTolerance = cupy.sqrt(1e-15) * n
    if blockVectorY is not None:
        if B is not None:
            blockVectorBY = B(blockVectorY)
        else:
            blockVectorBY = blockVectorY
        gramYBY = cupy.dot(blockVectorY.T.conj(), blockVectorBY)
        _applyConstraints(blockVectorX, gramYBY, blockVectorBY, blockVectorY)
    blockVectorX, blockVectorBX = _b_orthonormalize(B, blockVectorX)
    blockVectorAX = A(blockVectorX)
    gramXAX = cupy.dot(blockVectorX.T.conj(), blockVectorAX)
    _lambda, eigBlockVector = _eigh(gramXAX)
    ii = _get_indx(_lambda, sizeX, largest)
    _lambda = _lambda[ii]
    eigBlockVector = cupy.asarray(eigBlockVector[:, ii])
    blockVectorX = cupy.dot(blockVectorX, eigBlockVector)
    blockVectorAX = cupy.dot(blockVectorAX, eigBlockVector)
    if B is not None:
        blockVectorBX = cupy.dot(blockVectorBX, eigBlockVector)
    activeMask = cupy.ones((sizeX,), dtype=bool)
    lambdaHistory = [_lambda]
    residualNormsHistory = []
    previousBlockSize = sizeX
    ident = cupy.eye(sizeX, dtype=A.dtype)
    ident0 = cupy.eye(sizeX, dtype=A.dtype)
    blockVectorP = None
    blockVectorAP = None
    blockVectorBP = None
    iterationNumber = -1
    restart = True
    explicitGramFlag = False
    while iterationNumber < maxiter:
        iterationNumber += 1
        if verbosityLevel > 0:
            print('-' * 50)
            print('iteration %d' % iterationNumber)
        if B is not None:
            aux = blockVectorBX * _lambda[cupy.newaxis, :]
        else:
            aux = blockVectorX * _lambda[cupy.newaxis, :]
        blockVectorR = blockVectorAX - aux
        aux = cupy.sum(blockVectorR.conj() * blockVectorR, 0)
        residualNorms = cupy.sqrt(aux)
        residualNormsHistory.append(residualNorms)
        ii = cupy.where(residualNorms > residualTolerance, True, False)
        activeMask = activeMask & ii
        if verbosityLevel > 2:
            print(activeMask)
        currentBlockSize = int(activeMask.sum())
        if currentBlockSize != previousBlockSize:
            previousBlockSize = currentBlockSize
            ident = cupy.eye(currentBlockSize, dtype=A.dtype)
        if currentBlockSize == 0:
            break
        if verbosityLevel > 0:
            print(f'current block size: {currentBlockSize}')
            print(f'eigenvalue(s):\n{_lambda}')
            print(f'residual norm(s):\n{residualNorms}')
        if verbosityLevel > 10:
            print(eigBlockVector)
        activeBlockVectorR = _as2d(blockVectorR[:, activeMask])
        if iterationNumber > 0:
            activeBlockVectorP = _as2d(blockVectorP[:, activeMask])
            activeBlockVectorAP = _as2d(blockVectorAP[:, activeMask])
            if B is not None:
                activeBlockVectorBP = _as2d(blockVectorBP[:, activeMask])
        if M is not None:
            activeBlockVectorR = M(activeBlockVectorR)
        if blockVectorY is not None:
            _applyConstraints(activeBlockVectorR, gramYBY, blockVectorBY, blockVectorY)
        if B is not None:
            activeBlockVectorR = activeBlockVectorR - cupy.matmul(blockVectorX, cupy.matmul(blockVectorBX.T.conj(), activeBlockVectorR))
        else:
            activeBlockVectorR = activeBlockVectorR - cupy.matmul(blockVectorX, cupy.matmul(blockVectorX.T.conj(), activeBlockVectorR))
        aux = _b_orthonormalize(B, activeBlockVectorR)
        activeBlockVectorR, activeBlockVectorBR = aux
        activeBlockVectorAR = A(activeBlockVectorR)
        if iterationNumber > 0:
            if B is not None:
                aux = _b_orthonormalize(B, activeBlockVectorP, activeBlockVectorBP, retInvR=True)
                activeBlockVectorP, activeBlockVectorBP, invR, normal = aux
            else:
                aux = _b_orthonormalize(B, activeBlockVectorP, retInvR=True)
                activeBlockVectorP, _, invR, normal = aux
            if activeBlockVectorP is not None:
                activeBlockVectorAP = activeBlockVectorAP / normal
                activeBlockVectorAP = cupy.dot(activeBlockVectorAP, invR)
                restart = False
            else:
                restart = True
        if activeBlockVectorAR.dtype == 'float32':
            myeps = 1
        elif activeBlockVectorR.dtype == 'float32':
            myeps = 0.0001
        else:
            myeps = 1e-08
        if residualNorms.max() > myeps and (not explicitGramFlag):
            explicitGramFlag = False
        else:
            explicitGramFlag = True
        if B is None:
            blockVectorBX = blockVectorX
            activeBlockVectorBR = activeBlockVectorR
            if not restart:
                activeBlockVectorBP = activeBlockVectorP
        gramXAR = cupy.dot(blockVectorX.T.conj(), activeBlockVectorAR)
        gramRAR = cupy.dot(activeBlockVectorR.T.conj(), activeBlockVectorAR)
        if explicitGramFlag:
            gramRAR = (gramRAR + gramRAR.T.conj()) / 2
            gramXAX = cupy.dot(blockVectorX.T.conj(), blockVectorAX)
            gramXAX = (gramXAX + gramXAX.T.conj()) / 2
            gramXBX = cupy.dot(blockVectorX.T.conj(), blockVectorBX)
            gramRBR = cupy.dot(activeBlockVectorR.T.conj(), activeBlockVectorBR)
            gramXBR = cupy.dot(blockVectorX.T.conj(), activeBlockVectorBR)
        else:
            gramXAX = cupy.diag(_lambda)
            gramXBX = ident0
            gramRBR = ident
            gramXBR = cupy.zeros((int(sizeX), int(currentBlockSize)), dtype=A.dtype)

        def _handle_gramA_gramB_verbosity(gramA, gramB):
            if verbosityLevel > 0:
                _report_nonhermitian(gramA, 'gramA')
                _report_nonhermitian(gramB, 'gramB')
            if verbosityLevel > 10:
                numpy.savetxt('gramA.txt', cupy.asnumpy(gramA))
                numpy.savetxt('gramB.txt', cupy.asnumpy(gramB))
        if not restart:
            gramXAP = cupy.dot(blockVectorX.T.conj(), activeBlockVectorAP)
            gramRAP = cupy.dot(activeBlockVectorR.T.conj(), activeBlockVectorAP)
            gramPAP = cupy.dot(activeBlockVectorP.T.conj(), activeBlockVectorAP)
            gramXBP = cupy.dot(blockVectorX.T.conj(), activeBlockVectorBP)
            gramRBP = cupy.dot(activeBlockVectorR.T.conj(), activeBlockVectorBP)
            if explicitGramFlag:
                gramPAP = (gramPAP + gramPAP.T.conj()) / 2
                gramPBP = cupy.dot(activeBlockVectorP.T.conj(), activeBlockVectorBP)
            else:
                gramPBP = ident
            gramA = _bmat([[gramXAX, gramXAR, gramXAP], [gramXAR.T.conj(), gramRAR, gramRAP], [gramXAP.T.conj(), gramRAP.T.conj(), gramPAP]])
            gramB = _bmat([[gramXBX, gramXBR, gramXBP], [gramXBR.T.conj(), gramRBR, gramRBP], [gramXBP.T.conj(), gramRBP.T.conj(), gramPBP]])
            _handle_gramA_gramB_verbosity(gramA, gramB)
            try:
                _lambda, eigBlockVector = _eigh(gramA, gramB)
            except numpy.linalg.LinAlgError:
                restart = True
        if restart:
            gramA = _bmat([[gramXAX, gramXAR], [gramXAR.T.conj(), gramRAR]])
            gramB = _bmat([[gramXBX, gramXBR], [gramXBR.T.conj(), gramRBR]])
            _handle_gramA_gramB_verbosity(gramA, gramB)
            try:
                _lambda, eigBlockVector = _eigh(gramA, gramB)
            except numpy.linalg.LinAlgError:
                raise ValueError('eigh has failed in lobpcg iterations')
        ii = _get_indx(_lambda, sizeX, largest)
        if verbosityLevel > 10:
            print(ii)
            print(_lambda)
        _lambda = _lambda[ii]
        eigBlockVector = eigBlockVector[:, ii]
        lambdaHistory.append(_lambda)
        if verbosityLevel > 10:
            print('lambda:', _lambda)
        if verbosityLevel > 10:
            print(eigBlockVector)
        if B is not None:
            if not restart:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:sizeX + currentBlockSize]
                eigBlockVectorP = eigBlockVector[sizeX + currentBlockSize:]
                pp = cupy.dot(activeBlockVectorR, eigBlockVectorR)
                pp += cupy.dot(activeBlockVectorP, eigBlockVectorP)
                app = cupy.dot(activeBlockVectorAR, eigBlockVectorR)
                app += cupy.dot(activeBlockVectorAP, eigBlockVectorP)
                bpp = cupy.dot(activeBlockVectorBR, eigBlockVectorR)
                bpp += cupy.dot(activeBlockVectorBP, eigBlockVectorP)
            else:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:]
                pp = cupy.dot(activeBlockVectorR, eigBlockVectorR)
                app = cupy.dot(activeBlockVectorAR, eigBlockVectorR)
                bpp = cupy.dot(activeBlockVectorBR, eigBlockVectorR)
            if verbosityLevel > 10:
                print(pp)
                print(app)
                print(bpp)
            blockVectorX = cupy.dot(blockVectorX, eigBlockVectorX) + pp
            blockVectorAX = cupy.dot(blockVectorAX, eigBlockVectorX) + app
            blockVectorBX = cupy.dot(blockVectorBX, eigBlockVectorX) + bpp
            blockVectorP, blockVectorAP, blockVectorBP = (pp, app, bpp)
        else:
            if not restart:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:sizeX + currentBlockSize]
                eigBlockVectorP = eigBlockVector[sizeX + currentBlockSize:]
                pp = cupy.dot(activeBlockVectorR, eigBlockVectorR)
                pp += cupy.dot(activeBlockVectorP, eigBlockVectorP)
                app = cupy.dot(activeBlockVectorAR, eigBlockVectorR)
                app += cupy.dot(activeBlockVectorAP, eigBlockVectorP)
            else:
                eigBlockVectorX = eigBlockVector[:sizeX]
                eigBlockVectorR = eigBlockVector[sizeX:]
                pp = cupy.dot(activeBlockVectorR, eigBlockVectorR)
                app = cupy.dot(activeBlockVectorAR, eigBlockVectorR)
            if verbosityLevel > 10:
                print(pp)
                print(app)
            blockVectorX = cupy.dot(blockVectorX, eigBlockVectorX) + pp
            blockVectorAX = cupy.dot(blockVectorAX, eigBlockVectorX) + app
            blockVectorP, blockVectorAP = (pp, app)
    if B is not None:
        aux = blockVectorBX * _lambda[cupy.newaxis, :]
    else:
        aux = blockVectorX * _lambda[cupy.newaxis, :]
    blockVectorR = blockVectorAX - aux
    aux = cupy.sum(blockVectorR.conj() * blockVectorR, 0)
    residualNorms = cupy.sqrt(aux)
    if verbosityLevel > 0:
        print(f'Final eigenvalue(s):\n{_lambda}')
        print(f'Final residual norm(s):\n{residualNorms}')
    if retLambdaHistory:
        if retResidualNormsHistory:
            return (_lambda, blockVectorX, lambdaHistory, residualNormsHistory)
        else:
            return (_lambda, blockVectorX, lambdaHistory)
    elif retResidualNormsHistory:
        return (_lambda, blockVectorX, residualNormsHistory)
    else:
        return (_lambda, blockVectorX)
import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
def kkt_qr(G, dims, A):
    """
    Solution of KKT equations with zero 1,1 block, by eliminating the
    equality constraints via a QR factorization, and solving the
    reduced KKT system by another QR factorization.
    
    Computes the QR factorization
    
        A' = [Q1, Q2] * [R1; 0]
    
    and returns a function that (1) computes the QR factorization 
    
        W^{-T} * G * Q2 = Q3 * R3
    
    (with columns of W^{-T}*G in packed storage), and (2) returns a 
    function for solving 
    
        [ 0    A'   G'    ]   [ ux ]   [ bx ]
        [ A    0    0     ] * [ uy ] = [ by ].
        [ G    0   -W'*W  ]   [ uz ]   [ bz ]
    
    A is p x n and G is N x n where N = dims['l'] + sum(dims['q']) + 
    sum( k**2 for k in dims['s'] ).
    """
    p, n = A.size
    cdim = dims['l'] + sum(dims['q']) + sum([k ** 2 for k in dims['s']])
    cdim_pckd = dims['l'] + sum(dims['q']) + sum([int(k * (k + 1) / 2) for k in dims['s']])
    if type(A) is matrix:
        QA = +A.T
    else:
        QA = matrix(A.T)
    tauA = matrix(0.0, (p, 1))
    lapack.geqrf(QA, tauA)
    Gs = matrix(0.0, (cdim, n))
    tauG = matrix(0.0, (n - p, 1))
    u = matrix(0.0, (cdim_pckd, 1))
    vv = matrix(0.0, (n, 1))
    w = matrix(0.0, (cdim_pckd, 1))

    def factor(W):
        Gs[:, :] = G
        scale(Gs, W, trans='T', inverse='I')
        pack2(Gs, dims)
        lapack.ormqr(QA, tauA, Gs, side='R', m=cdim_pckd)
        lapack.geqrf(Gs, tauG, n=n - p, m=cdim_pckd, offsetA=Gs.size[0] * p)

        def solve(x, y, z):
            scale(z, W, trans='T', inverse='I')
            pack(z, w, dims)
            blas.copy(x, vv)
            lapack.ormqr(QA, tauA, vv, trans='T')
            lapack.trtrs(Gs, vv, uplo='U', trans='T', n=n - p, offsetA=Gs.size[0] * p, offsetB=p)
            blas.copy(y, x)
            lapack.trtrs(QA, x, uplo='U', trans='T', n=p)
            blas.gemv(Gs, x, w, alpha=-1.0, beta=1.0, n=p, m=cdim_pckd)
            blas.copy(w, u)
            lapack.ormqr(Gs, tauG, u, trans='T', k=n - p, offsetA=Gs.size[0] * p, m=cdim_pckd)
            blas.axpy(vv, u, offsetx=p, n=n - p)
            blas.scal(0.0, u, offset=n - p)
            blas.copy(u, x, offsety=p, n=n - p)
            lapack.trtrs(Gs, x, uplo='U', n=n - p, offsetA=Gs.size[0] * p, offsetB=p)
            lapack.ormqr(QA, tauA, x)
            lapack.ormqr(Gs, tauG, u, k=n - p, m=cdim_pckd, offsetA=Gs.size[0] * p)
            blas.axpy(w, u, alpha=-1.0)
            blas.copy(vv, y, n=p)
            blas.gemv(Gs, u, y, m=cdim_pckd, n=p, trans='T', alpha=-1.0, beta=1.0)
            lapack.trtrs(QA, y, uplo='U', n=p)
            unpack(u, z, dims)
        return solve
    return factor
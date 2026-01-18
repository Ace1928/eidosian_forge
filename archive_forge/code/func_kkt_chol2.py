import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
def kkt_chol2(G, dims, A, mnl=0):
    """
    Solution of KKT equations by reduction to a 2 x 2 system, a sparse 
    or dense Cholesky factorization of order n to eliminate the 1,1 
    block, and a sparse or dense Cholesky factorization of order p.
    Implemented only for problems with no second-order or semidefinite
    cone constraints.
    
    Returns a function that (1) computes Cholesky factorizations of
    the matrices 
    
        S = H + GG' * W^{-1} * W^{-T} * GG,  
        K = A * S^{-1} *A'
    
    or (if K is singular in the first call to the function), the matrices
    
        S = H + GG' * W^{-1} * W^{-T} * GG + A' * A,  
        K = A * S^{-1} * A',
    
    given H, Df, W, where GG = [Df; G], and (2) returns a function for 
    solving 
    
        [ H     A'   GG'   ]   [ ux ]   [ bx ]
        [ A     0    0     ] * [ uy ] = [ by ].
        [ GG    0   -W'*W  ]   [ uz ]   [ bz ]
    
    H is n x n,  A is p x n, Df is mnl x n, G is dims['l'] x n.
    """
    if dims['q'] or dims['s']:
        raise ValueError("kktsolver option 'kkt_chol2' is implemented only for problems with no second-order or semidefinite cone constraints")
    p, n = A.size
    ml = dims['l']
    F = {'firstcall': True, 'singular': False}

    def factor(W, H=None, Df=None):
        if F['firstcall']:
            if type(G) is matrix:
                F['Gs'] = matrix(0.0, G.size)
            else:
                F['Gs'] = spmatrix(0.0, G.I, G.J, G.size)
            if mnl:
                if type(Df) is matrix:
                    F['Dfs'] = matrix(0.0, Df.size)
                else:
                    F['Dfs'] = spmatrix(0.0, Df.I, Df.J, Df.size)
            if mnl and type(Df) is matrix or type(G) is matrix or type(H) is matrix:
                F['S'] = matrix(0.0, (n, n))
                F['K'] = matrix(0.0, (p, p))
            else:
                F['S'] = spmatrix([], [], [], (n, n), 'd')
                F['Sf'] = None
                if type(A) is matrix:
                    F['K'] = matrix(0.0, (p, p))
                else:
                    F['K'] = spmatrix([], [], [], (p, p), 'd')
        if mnl:
            base.gemm(spmatrix(W['dnli'], list(range(mnl)), list(range(mnl))), Df, F['Dfs'], partial=True)
        base.gemm(spmatrix(W['di'], list(range(ml)), list(range(ml))), G, F['Gs'], partial=True)
        if F['firstcall']:
            base.syrk(F['Gs'], F['S'], trans='T')
            if mnl:
                base.syrk(F['Dfs'], F['S'], trans='T', beta=1.0)
            if H is not None:
                F['S'] += H
            try:
                if type(F['S']) is matrix:
                    lapack.potrf(F['S'])
                else:
                    F['Sf'] = cholmod.symbolic(F['S'])
                    cholmod.numeric(F['S'], F['Sf'])
            except ArithmeticError:
                F['singular'] = True
                if type(A) is matrix and type(F['S']) is spmatrix:
                    F['S'] = matrix(0.0, (n, n))
                base.syrk(F['Gs'], F['S'], trans='T')
                if mnl:
                    base.syrk(F['Dfs'], F['S'], trans='T', beta=1.0)
                base.syrk(A, F['S'], trans='T', beta=1.0)
                if H is not None:
                    F['S'] += H
                if type(F['S']) is matrix:
                    lapack.potrf(F['S'])
                else:
                    F['Sf'] = cholmod.symbolic(F['S'])
                    cholmod.numeric(F['S'], F['Sf'])
            F['firstcall'] = False
        else:
            base.syrk(F['Gs'], F['S'], trans='T', partial=True)
            if mnl:
                base.syrk(F['Dfs'], F['S'], trans='T', beta=1.0, partial=True)
            if H is not None:
                F['S'] += H
            if F['singular']:
                base.syrk(A, F['S'], trans='T', beta=1.0, partial=True)
            if type(F['S']) is matrix:
                lapack.potrf(F['S'])
            else:
                cholmod.numeric(F['S'], F['Sf'])
        if type(F['S']) is matrix:
            if type(A) is matrix:
                Asct = A.T
            else:
                Asct = matrix(A.T)
            blas.trsm(F['S'], Asct)
            blas.syrk(Asct, F['K'], trans='T')
            lapack.potrf(F['K'])
        elif type(A) is matrix:
            Asct = A.T
            cholmod.solve(F['Sf'], Asct, sys=7)
            cholmod.solve(F['Sf'], Asct, sys=4)
            blas.syrk(Asct, F['K'], trans='T')
            lapack.potrf(F['K'])
        else:
            Asct = cholmod.spsolve(F['Sf'], A.T, sys=7)
            Asct = cholmod.spsolve(F['Sf'], Asct, sys=4)
            base.syrk(Asct, F['K'], trans='T')
            Kf = cholmod.symbolic(F['K'])
            cholmod.numeric(F['K'], Kf)

        def solve(x, y, z):
            scale(z, W, trans='T', inverse='I')
            if mnl:
                base.gemv(F['Dfs'], z, x, trans='T', beta=1.0)
            base.gemv(F['Gs'], z, x, offsetx=mnl, trans='T', beta=1.0)
            if F['singular']:
                base.gemv(A, y, x, trans='T', beta=1.0)
            if type(F['S']) is matrix:
                blas.trsv(F['S'], x)
            else:
                cholmod.solve(F['Sf'], x, sys=7)
                cholmod.solve(F['Sf'], x, sys=4)
            base.gemv(Asct, x, y, trans='T', beta=-1.0)
            if type(F['K']) is matrix:
                lapack.potrs(F['K'], y)
            else:
                cholmod.solve(Kf, y)
            base.gemv(Asct, y, x, alpha=-1.0, beta=1.0)
            if type(F['S']) is matrix:
                blas.trsv(F['S'], x, trans='T')
            else:
                cholmod.solve(F['Sf'], x, sys=5)
                cholmod.solve(F['Sf'], x, sys=8)
            if mnl:
                base.gemv(F['Dfs'], x, z, beta=-1.0)
            base.gemv(F['Gs'], x, z, beta=-1.0, offsety=mnl)
        return solve
    return factor
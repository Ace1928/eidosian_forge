import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
def sgemv(A, x, y, dims, trans='N', alpha=1.0, beta=0.0, n=None, offsetA=0, offsetx=0, offsety=0):
    """
    Matrix-vector multiplication.

    A is a matrix or spmatrix of size (m, n) where 
    
        N = dims['l'] + sum(dims['q']) + sum( k**2 for k in dims['s'] ) 

    representing a mapping from R^n to S.  
    
    If trans is 'N': 
    
        y := alpha*A*x + beta * y   (trans = 'N').
    
    x is a vector of length n.  y is a vector of length N.
    
    If trans is 'T':
    
        y := alpha*A'*x + beta * y  (trans = 'T').
    
    x is a vector of length N.  y is a vector of length n.
    
    The 's' components in S are stored in unpacked 'L' storage.
    """
    m = dims['l'] + sum(dims['q']) + sum([k ** 2 for k in dims['s']])
    if n is None:
        n = A.size[1]
    if trans == 'T' and alpha:
        trisc(x, dims, offsetx)
    base.gemv(A, x, y, trans=trans, alpha=alpha, beta=beta, m=m, n=n, offsetA=offsetA, offsetx=offsetx, offsety=offsety)
    if trans == 'T' and alpha:
        triusc(x, dims, offsetx)
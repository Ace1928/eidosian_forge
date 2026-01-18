import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
def scale2(lmbda, x, dims, mnl=0, inverse='N'):
    """
    Evaluates

        x := H(lambda^{1/2}) * x   (inverse is 'N')
        x := H(lambda^{-1/2}) * x  (inverse is 'I').
    
    H is the Hessian of the logarithmic barrier.
    """
    if inverse == 'N':
        blas.tbsv(lmbda, x, n=mnl + dims['l'], k=0, ldA=1)
    else:
        blas.tbmv(lmbda, x, n=mnl + dims['l'], k=0, ldA=1)
    ind = mnl + dims['l']
    for m in dims['q']:
        a = jnrm2(lmbda, n=m, offset=ind)
        if inverse == 'N':
            lx = jdot(lmbda, x, n=m, offsetx=ind, offsety=ind) / a
        else:
            lx = blas.dot(lmbda, x, n=m, offsetx=ind, offsety=ind) / a
        x0 = x[ind]
        x[ind] = lx
        c = (lx + x0) / (lmbda[ind] / a + 1) / a
        if inverse == 'N':
            c *= -1.0
        blas.axpy(lmbda, x, alpha=c, n=m - 1, offsetx=ind + 1, offsety=ind + 1)
        if inverse == 'N':
            a = 1.0 / a
        blas.scal(a, x, offset=ind, n=m)
        ind += m
    ind2 = ind
    for k in range(len(dims['s'])):
        m = dims['s'][k]
        for j in range(m):
            c = math.sqrt(lmbda[ind2 + j]) * base.sqrt(lmbda[ind2:ind2 + m])
            if inverse == 'N':
                blas.tbsv(c, x, n=m, k=0, ldA=1, offsetx=ind + j * m)
            else:
                blas.tbmv(c, x, n=m, k=0, ldA=1, offsetx=ind + j * m)
        ind += m * m
        ind2 += m
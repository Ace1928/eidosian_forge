import math
from cvxopt import base, blas, lapack, cholmod, misc_solvers
from cvxopt.base import matrix, spmatrix
def compute_scaling(s, z, lmbda, dims, mnl=None):
    """
    Returns the Nesterov-Todd scaling W at points s and z, and stores the 
    scaled variable in lmbda. 
    
        W * z = W^{-T} * s = lmbda. 

    """
    W = {}
    if mnl is None:
        mnl = 0
    else:
        W['dnl'] = base.sqrt(base.div(s[:mnl], z[:mnl]))
        W['dnli'] = W['dnl'] ** (-1)
        lmbda[:mnl] = base.sqrt(base.mul(s[:mnl], z[:mnl]))
    m = dims['l']
    W['d'] = base.sqrt(base.div(s[mnl:mnl + m], z[mnl:mnl + m]))
    W['di'] = W['d'] ** (-1)
    lmbda[mnl:mnl + m] = base.sqrt(base.mul(s[mnl:mnl + m], z[mnl:mnl + m]))
    ind = mnl + dims['l']
    W['v'] = [matrix(0.0, (k, 1)) for k in dims['q']]
    W['beta'] = len(dims['q']) * [0.0]
    for k in range(len(dims['q'])):
        m = dims['q'][k]
        v = W['v'][k]
        aa = jnrm2(s, offset=ind, n=m)
        bb = jnrm2(z, offset=ind, n=m)
        W['beta'][k] = math.sqrt(aa / bb)
        cc = math.sqrt((blas.dot(s, z, n=m, offsetx=ind, offsety=ind) / aa / bb + 1.0) / 2.0)
        blas.copy(z, v, offsetx=ind, n=m)
        blas.scal(-1.0 / bb, v)
        v[0] *= -1.0
        blas.axpy(s, v, 1.0 / aa, offsetx=ind, n=m)
        blas.scal(1.0 / 2.0 / cc, v)
        v[0] += 1.0
        blas.scal(1.0 / math.sqrt(2.0 * v[0]), v)
        lmbda[ind] = cc
        dd = 2 * cc + s[ind] / aa + z[ind] / bb
        blas.copy(s, lmbda, offsetx=ind + 1, offsety=ind + 1, n=m - 1)
        blas.scal((cc + z[ind] / bb) / dd / aa, lmbda, n=m - 1, offset=ind + 1)
        blas.axpy(z, lmbda, (cc + s[ind] / aa) / dd / bb, n=m - 1, offsetx=ind + 1, offsety=ind + 1)
        blas.scal(math.sqrt(aa * bb), lmbda, offset=ind, n=m)
        ind += m
    W['r'] = [matrix(0.0, (m, m)) for m in dims['s']]
    W['rti'] = [matrix(0.0, (m, m)) for m in dims['s']]
    work = matrix(0.0, (max([0] + dims['s']) ** 2, 1))
    Ls = matrix(0.0, (max([0] + dims['s']) ** 2, 1))
    Lz = matrix(0.0, (max([0] + dims['s']) ** 2, 1))
    ind2 = ind
    for k in range(len(dims['s'])):
        m = dims['s'][k]
        r, rti = (W['r'][k], W['rti'][k])
        blas.copy(s, Ls, offsetx=ind2, n=m ** 2)
        lapack.potrf(Ls, n=m, ldA=m)
        blas.copy(z, Lz, offsetx=ind2, n=m ** 2)
        lapack.potrf(Lz, n=m, ldA=m)
        for i in range(m):
            blas.scal(0.0, Ls, offset=i * m, n=i)
        blas.copy(Ls, work, n=m ** 2)
        blas.trmm(Lz, work, transA='T', ldA=m, ldB=m, n=m, m=m)
        lapack.gesvd(work, lmbda, jobu='O', ldA=m, m=m, n=m, offsetS=ind)
        blas.copy(work, r, n=m * m)
        blas.trsm(Lz, r, transA='T', m=m, n=m, ldA=m)
        blas.copy(work, rti, n=m * m)
        blas.trmm(Lz, rti, m=m, n=m, ldA=m)
        for i in range(m):
            a = math.sqrt(lmbda[ind + i])
            blas.scal(a, r, offset=m * i, n=m)
            blas.scal(1.0 / a, rti, offset=m * i, n=m)
        ind += m
        ind2 += m * m
    return W
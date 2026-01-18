import sys
def f4_no_ir(x, y, z, s):
    misc.sinv(s, lmbda, dims, mnl)
    blas.copy(s, ws3)
    misc.scale(ws3, W, trans='T')
    blas.axpy(ws3, z, alpha=-1.0)
    f3(x, y, z)
    blas.axpy(z, s, alpha=-1.0)
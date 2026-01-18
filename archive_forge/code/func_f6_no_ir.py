import sys
def f6_no_ir(x, y, z, tau, s, kappa):
    yscal(-1.0, y)
    misc.sinv(s, lmbda, dims)
    blas.scal(-1.0, s)
    blas.copy(s, ws3)
    misc.scale(ws3, W, trans='T')
    blas.axpy(ws3, z)
    blas.scal(-1.0, z)
    f3(x, y, z)
    kappa[0] = -kappa[0] / lmbda[-1]
    tau[0] += kappa[0] / dgi
    tau[0] = dgi * (tau[0] + xdot(c, x) + ydot(b, y) + misc.sdot(th, z, dims)) / (1.0 + misc.sdot(z1, z1, dims))
    xaxpy(x1, x, alpha=tau[0])
    yaxpy(y1, y, alpha=tau[0])
    blas.axpy(z1, z, alpha=tau[0])
    blas.axpy(z, s, alpha=-1)
    kappa[0] -= tau[0]
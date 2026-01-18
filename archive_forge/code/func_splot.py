from colorsys import hsv_to_rgb, hls_to_rgb
from .libmp import NoConvergence
from .libmp.backend import xrange
def splot(ctx, f, u=[-5, 5], v=[-5, 5], points=100, keep_aspect=True, wireframe=False, file=None, dpi=None, axes=None):
    """
    Plots the surface defined by `f`.

    If `f` returns a single component, then this plots the surface
    defined by `z = f(x,y)` over the rectangular domain with
    `x = u` and `y = v`.

    If `f` returns three components, then this plots the parametric
    surface `x, y, z = f(u,v)` over the pairs of intervals `u` and `v`.

    For example, to plot a simple function::

        >>> from mpmath import *
        >>> f = lambda x, y: sin(x+y)*cos(y)
        >>> splot(f, [-pi,pi], [-pi,pi])    # doctest: +SKIP

    Plotting a donut::

        >>> r, R = 1, 2.5
        >>> f = lambda u, v: [r*cos(u), (R+r*sin(u))*cos(v), (R+r*sin(u))*sin(v)]
        >>> splot(f, [0, 2*pi], [0, 2*pi])    # doctest: +SKIP

    .. note :: This function requires matplotlib (pylab) 0.98.5.3 or higher.
    """
    import pylab
    import mpl_toolkits.mplot3d as mplot3d
    if file:
        axes = None
    fig = None
    if not axes:
        fig = pylab.figure()
        axes = mplot3d.axes3d.Axes3D(fig)
    ua, ub = u
    va, vb = v
    du = ub - ua
    dv = vb - va
    if not isinstance(points, (list, tuple)):
        points = [points, points]
    M, N = points
    u = pylab.linspace(ua, ub, M)
    v = pylab.linspace(va, vb, N)
    x, y, z = [pylab.zeros((M, N)) for i in xrange(3)]
    xab, yab, zab = [[0, 0] for i in xrange(3)]
    for n in xrange(N):
        for m in xrange(M):
            fdata = f(ctx.convert(u[m]), ctx.convert(v[n]))
            try:
                x[m, n], y[m, n], z[m, n] = fdata
            except TypeError:
                x[m, n], y[m, n], z[m, n] = (u[m], v[n], fdata)
            for c, cab in [(x[m, n], xab), (y[m, n], yab), (z[m, n], zab)]:
                if c < cab[0]:
                    cab[0] = c
                if c > cab[1]:
                    cab[1] = c
    if wireframe:
        axes.plot_wireframe(x, y, z, rstride=4, cstride=4)
    else:
        axes.plot_surface(x, y, z, rstride=4, cstride=4)
    axes.set_xlabel('x')
    axes.set_ylabel('y')
    axes.set_zlabel('z')
    if keep_aspect:
        dx, dy, dz = [cab[1] - cab[0] for cab in [xab, yab, zab]]
        maxd = max(dx, dy, dz)
        if dx < maxd:
            delta = maxd - dx
            axes.set_xlim3d(xab[0] - delta / 2.0, xab[1] + delta / 2.0)
        if dy < maxd:
            delta = maxd - dy
            axes.set_ylim3d(yab[0] - delta / 2.0, yab[1] + delta / 2.0)
        if dz < maxd:
            delta = maxd - dz
            axes.set_zlim3d(zab[0] - delta / 2.0, zab[1] + delta / 2.0)
    if fig:
        if file:
            pylab.savefig(file, dpi=dpi)
        else:
            pylab.show()
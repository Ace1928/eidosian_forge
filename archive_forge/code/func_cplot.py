from colorsys import hsv_to_rgb, hls_to_rgb
from .libmp import NoConvergence
from .libmp.backend import xrange
def cplot(ctx, f, re=[-5, 5], im=[-5, 5], points=2000, color=None, verbose=False, file=None, dpi=None, axes=None):
    """
    Plots the given complex-valued function *f* over a rectangular part
    of the complex plane specified by the pairs of intervals *re* and *im*.
    For example::

        cplot(lambda z: z, [-2, 2], [-10, 10])
        cplot(exp)
        cplot(zeta, [0, 1], [0, 50])

    By default, the complex argument (phase) is shown as color (hue) and
    the magnitude is show as brightness. You can also supply a
    custom color function (*color*). This function should take a
    complex number as input and return an RGB 3-tuple containing
    floats in the range 0.0-1.0.

    Alternatively, you can select a builtin color function by passing
    a string as *color*:

      * "default" - default color scheme
      * "phase" - a color scheme that only renders the phase of the function,
         with white for positive reals, black for negative reals, gold in the
         upper half plane, and blue in the lower half plane.

    To obtain a sharp image, the number of points may need to be
    increased to 100,000 or thereabout. Since evaluating the
    function that many times is likely to be slow, the 'verbose'
    option is useful to display progress.

    .. note :: This function requires matplotlib (pylab).
    """
    if color is None or color == 'default':
        color = ctx.default_color_function
    if color == 'phase':
        color = ctx.phase_color_function
    import pylab
    if file:
        axes = None
    fig = None
    if not axes:
        fig = pylab.figure()
        axes = fig.add_subplot(111)
    rea, reb = re
    ima, imb = im
    dre = reb - rea
    dim = imb - ima
    M = int(ctx.sqrt(points * dre / dim) + 1)
    N = int(ctx.sqrt(points * dim / dre) + 1)
    x = pylab.linspace(rea, reb, M)
    y = pylab.linspace(ima, imb, N)
    w = pylab.zeros((N, M, 3))
    for n in xrange(N):
        for m in xrange(M):
            z = ctx.mpc(x[m], y[n])
            try:
                v = color(f(z))
            except ctx.plot_ignore:
                v = (0.5, 0.5, 0.5)
            w[n, m] = v
        if verbose:
            print(str(n) + ' of ' + str(N))
    rea, reb, ima, imb = [float(_) for _ in [rea, reb, ima, imb]]
    axes.imshow(w, extent=(rea, reb, ima, imb), origin='lower')
    axes.set_xlabel('Re(z)')
    axes.set_ylabel('Im(z)')
    if fig:
        if file:
            pylab.savefig(file, dpi=dpi)
        else:
            pylab.show()
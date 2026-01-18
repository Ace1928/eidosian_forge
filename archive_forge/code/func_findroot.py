from __future__ import print_function
from copy import copy
from ..libmp.backend import xrange
def findroot(ctx, f, x0, solver='secant', tol=None, verbose=False, verify=True, **kwargs):
    """
    Find an approximate solution to `f(x) = 0`, using *x0* as starting point or
    interval for *x*.

    Multidimensional overdetermined systems are supported.
    You can specify them using a function or a list of functions.

    Mathematically speaking, this function returns `x` such that
    `|f(x)|^2 \\leq \\mathrm{tol}` is true within the current working precision.
    If the computed value does not meet this criterion, an exception is raised.
    This exception can be disabled with *verify=False*.

    For interval arithmetic (``iv.findroot()``), please note that
    the returned interval ``x`` is not guaranteed to contain `f(x)=0`!
    It is only some `x` for which `|f(x)|^2 \\leq \\mathrm{tol}` certainly holds
    regardless of numerical error. This may be improved in the future.

    **Arguments**

    *f*
        one dimensional function
    *x0*
        starting point, several starting points or interval (depends on solver)
    *tol*
        the returned solution has an error smaller than this
    *verbose*
        print additional information for each iteration if true
    *verify*
        verify the solution and raise a ValueError if `|f(x)|^2 > \\mathrm{tol}`
    *solver*
        a generator for *f* and *x0* returning approximative solution and error
    *maxsteps*
        after how many steps the solver will cancel
    *df*
        first derivative of *f* (used by some solvers)
    *d2f*
        second derivative of *f* (used by some solvers)
    *multidimensional*
        force multidimensional solving
    *J*
        Jacobian matrix of *f* (used by multidimensional solvers)
    *norm*
        used vector norm (used by multidimensional solvers)

    solver has to be callable with ``(f, x0, **kwargs)`` and return an generator
    yielding pairs of approximative solution and estimated error (which is
    expected to be positive).
    You can use the following string aliases:
    'secant', 'mnewton', 'halley', 'muller', 'illinois', 'pegasus', 'anderson',
    'ridder', 'anewton', 'bisect'

    See mpmath.calculus.optimization for their documentation.

    **Examples**

    The function :func:`~mpmath.findroot` locates a root of a given function using the
    secant method by default. A simple example use of the secant method is to
    compute `\\pi` as the root of `\\sin x` closest to `x_0 = 3`::

        >>> from mpmath import *
        >>> mp.dps = 30; mp.pretty = True
        >>> findroot(sin, 3)
        3.14159265358979323846264338328

    The secant method can be used to find complex roots of analytic functions,
    although it must in that case generally be given a nonreal starting value
    (or else it will never leave the real line)::

        >>> mp.dps = 15
        >>> findroot(lambda x: x**3 + 2*x + 1, j)
        (0.226698825758202 + 1.46771150871022j)

    A nice application is to compute nontrivial roots of the Riemann zeta
    function with many digits (good initial values are needed for convergence)::

        >>> mp.dps = 30
        >>> findroot(zeta, 0.5+14j)
        (0.5 + 14.1347251417346937904572519836j)

    The secant method can also be used as an optimization algorithm, by passing
    it a derivative of a function. The following example locates the positive
    minimum of the gamma function::

        >>> mp.dps = 20
        >>> findroot(lambda x: diff(gamma, x), 1)
        1.4616321449683623413

    Finally, a useful application is to compute inverse functions, such as the
    Lambert W function which is the inverse of `w e^w`, given the first
    term of the solution's asymptotic expansion as the initial value. In basic
    cases, this gives identical results to mpmath's built-in ``lambertw``
    function::

        >>> def lambert(x):
        ...     return findroot(lambda w: w*exp(w) - x, log(1+x))
        ...
        >>> mp.dps = 15
        >>> lambert(1); lambertw(1)
        0.567143290409784
        0.567143290409784
        >>> lambert(1000); lambert(1000)
        5.2496028524016
        5.2496028524016

    Multidimensional functions are also supported::

        >>> f = [lambda x1, x2: x1**2 + x2,
        ...      lambda x1, x2: 5*x1**2 - 3*x1 + 2*x2 - 3]
        >>> findroot(f, (0, 0))
        [-0.618033988749895]
        [-0.381966011250105]
        >>> findroot(f, (10, 10))
        [ 1.61803398874989]
        [-2.61803398874989]

    You can verify this by solving the system manually.

    Please note that the following (more general) syntax also works::

        >>> def f(x1, x2):
        ...     return x1**2 + x2, 5*x1**2 - 3*x1 + 2*x2 - 3
        ...
        >>> findroot(f, (0, 0))
        [-0.618033988749895]
        [-0.381966011250105]


    **Multiple roots**

    For multiple roots all methods of the Newtonian family (including secant)
    converge slowly. Consider this example::

        >>> f = lambda x: (x - 1)**99
        >>> findroot(f, 0.9, verify=False)
        0.918073542444929

    Even for a very close starting point the secant method converges very
    slowly. Use ``verbose=True`` to illustrate this.

    It is possible to modify Newton's method to make it converge regardless of
    the root's multiplicity::

        >>> findroot(f, -10, solver='mnewton')
        1.0

    This variant uses the first and second derivative of the function, which is
    not very efficient.

    Alternatively you can use an experimental Newtonian solver that keeps track
    of the speed of convergence and accelerates it using Steffensen's method if
    necessary::

        >>> findroot(f, -10, solver='anewton', verbose=True)
        x:     -9.88888888888888888889
        error: 0.111111111111111111111
        converging slowly
        x:     -9.77890011223344556678
        error: 0.10998877665544332211
        converging slowly
        x:     -9.67002233332199662166
        error: 0.108877778911448945119
        converging slowly
        accelerating convergence
        x:     -9.5622443299551077669
        error: 0.107778003366888854764
        converging slowly
        x:     0.99999999999999999214
        error: 10.562244329955107759
        x:     1.0
        error: 7.8598304758094664213e-18
        ZeroDivisionError: canceled with x = 1.0
        1.0

    **Complex roots**

    For complex roots it's recommended to use Muller's method as it converges
    even for real starting points very fast::

        >>> findroot(lambda x: x**4 + x + 1, (0, 1, 2), solver='muller')
        (0.727136084491197 + 0.934099289460529j)


    **Intersection methods**

    When you need to find a root in a known interval, it's highly recommended to
    use an intersection-based solver like ``'anderson'`` or ``'ridder'``.
    Usually they converge faster and more reliable. They have however problems
    with multiple roots and usually need a sign change to find a root::

        >>> findroot(lambda x: x**3, (-1, 1), solver='anderson')
        0.0

    Be careful with symmetric functions::

        >>> findroot(lambda x: x**2, (-1, 1), solver='anderson') #doctest:+ELLIPSIS
        Traceback (most recent call last):
          ...
        ZeroDivisionError

    It fails even for better starting points, because there is no sign change::

        >>> findroot(lambda x: x**2, (-1, .5), solver='anderson')
        Traceback (most recent call last):
          ...
        ValueError: Could not find root within given tolerance. (1.0 > 2.16840434497100886801e-19)
        Try another starting point or tweak arguments.

    """
    prec = ctx.prec
    try:
        ctx.prec += 20
        if tol is None:
            tol = ctx.eps * 2 ** 10
        kwargs['verbose'] = kwargs.get('verbose', verbose)
        if 'd1f' in kwargs:
            kwargs['df'] = kwargs['d1f']
        kwargs['tol'] = tol
        if isinstance(x0, (list, tuple)):
            x0 = [ctx.convert(x) for x in x0]
        else:
            x0 = [ctx.convert(x0)]
        if isinstance(solver, str):
            try:
                solver = str2solver[solver]
            except KeyError:
                raise ValueError('could not recognize solver')
        if isinstance(f, (list, tuple)):
            f2 = copy(f)

            def tmp(*args):
                return [fn(*args) for fn in f2]
            f = tmp
        try:
            fx = f(*x0)
            multidimensional = isinstance(fx, (list, tuple, ctx.matrix))
        except TypeError:
            fx = f(x0[0])
            multidimensional = False
        if 'multidimensional' in kwargs:
            multidimensional = kwargs['multidimensional']
        if multidimensional:
            solver = MDNewton
            if not 'norm' in kwargs:
                norm = lambda x: ctx.norm(x, 'inf')
                kwargs['norm'] = norm
            else:
                norm = kwargs['norm']
        else:
            norm = abs
        if norm(fx) == 0:
            if multidimensional:
                return ctx.matrix(x0)
            else:
                return x0[0]
        iterations = solver(ctx, f, x0, **kwargs)
        if 'maxsteps' in kwargs:
            maxsteps = kwargs['maxsteps']
        else:
            maxsteps = iterations.maxsteps
        i = 0
        for x, error in iterations:
            if verbose:
                print('x:    ', x)
                print('error:', error)
            i += 1
            if error < tol * max(1, norm(x)) or i >= maxsteps:
                break
        else:
            if not i:
                raise ValueError('Could not find root using the given solver.\nTry another starting point or tweak arguments.')
        if not isinstance(x, (list, tuple, ctx.matrix)):
            xl = [x]
        else:
            xl = x
        if verify and norm(f(*xl)) ** 2 > tol:
            raise ValueError('Could not find root within given tolerance. (%s > %s)\nTry another starting point or tweak arguments.' % (norm(f(*xl)) ** 2, tol))
        return x
    finally:
        ctx.prec = prec
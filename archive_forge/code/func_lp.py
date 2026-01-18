import sys
def lp(c, G, h, A=None, b=None, kktsolver=None, solver=None, primalstart=None, dualstart=None, **kwargs):
    """

    Solves a pair of primal and dual LPs

        minimize    c'*x
        subject to  G*x + s = h
                    A*x = b
                    s >= 0

        maximize    -h'*z - b'*y
        subject to  G'*z + A'*y + c = 0
                    z >= 0.


    Input arguments.

        c is n x 1, G is m x n, h is m x 1, A is p x n, b is p x 1.  G and
        A must be dense or sparse 'd' matrices.  c, h and b are dense 'd'
        matrices with one column.  The default values for A and b are
        empty matrices with zero rows.

        solver is None, 'glpk' or 'mosek'.  The default solver (None)
        uses the cvxopt conelp() function.  The 'glpk' solver is the
        simplex LP solver from GLPK.  The 'mosek' solver is the LP solver
        from MOSEK.

        The arguments primalstart and dualstart are ignored when solver
        is 'glpk' or 'mosek', and are optional when solver is None.
        The argument primalstart is a dictionary with keys 'x' and 's',
        and specifies a primal starting point.  primalstart['x'] must
        be a dense 'd' matrix of length n;  primalstart['s'] must be a
        positive dense 'd' matrix of length m.
        The argument dualstart is a dictionary with keys 'z' and 'y',
        and specifies a dual starting point.   dualstart['y'] must
        be a dense 'd' matrix of length p;  dualstart['z'] must be a
        positive dense 'd' matrix of length m.

        When solver is None, we require n >= 1, Rank(A) = p and
        Rank([G; A]) = n


    Output arguments.

        Returns a dictionary with keys 'status', 'x', 's', 'z', 'y',
        'primal objective', 'dual objective', 'gap', 'relative gap',
        'primal infeasibility', 'dual infeasibility', 'primal slack',
        'dual slack', 'residual as primal infeasibility certificate',
        'residual as dual infeasibility certificate'.

        The 'status' field has values 'optimal', 'primal infeasible',
        'dual infeasible', or 'unknown'.  The values of the other fields
        depend on the exit status and the solver used.

        Status 'optimal'.
        - 'x', 's', 'y', 'z' are an approximate solution of the primal and
          dual optimality conditions

              G*x + s = h,  A*x = b
              G'*z + A'*y + c = 0
              s >= 0, z >= 0
              s'*z = 0.

        - 'primal objective': the primal objective c'*x.
        - 'dual objective': the dual objective -h'*z - b'*y.
        - 'gap': the duality gap s'*z.
        - 'relative gap': the relative gap, defined as s'*z / -c'*x if
          the primal objective is negative, s'*z / -(h'*z + b'*y) if the
          dual objective is positive, and None otherwise.
        - 'primal infeasibility': the residual in the primal constraints,
          defined as the maximum of the residual in the inequalities

              || G*x + s - h || / max(1, ||h||)

          and the residual in the equalities

              || A*x - b || / max(1, ||b||).

        - 'dual infeasibility': the residual in the dual constraints,
          defined as

              || G'*z + A'*y + c || / max(1, ||c||).

        - 'primal slack': the smallest primal slack min_k s_k.
        - 'dual slack': the smallest dual slack min_k z_k.
        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate': None.
        If the default solver is used, the primal infeasibility is
        guaranteed to be less than solvers.options['feastol']
        (default 1e-7).  The dual infeasibility is guaranteed to be less
        than solvers.options['feastol'] (default 1e-7).  The gap is less
        than solvers.options['abstol'] (default 1e-7) or the relative gap
        is less than solvers.options['reltol'] (default 1e-6).
        For the other solvers, the default GLPK or MOSEK exit criteria
        apply.

        Status 'primal infeasible'.  If the GLPK solver is used, all the
        fields except the status field are None.  For the default and
        the MOSEK solvers, the values are as follows.
        - 'x', 's': None.
        - 'y', 'z' are an approximate certificate of infeasibility

              -h'*z - b'*y = 1,  G'*z + A'*y = 0,  z >= 0.

        - 'primal objective': None.
        - 'dual objective': 1.0.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': None.
        - 'dual slack': the smallest dual slack min z_k.
        - 'residual as primal infeasibility certificate': the residual in
          the condition of the infeasibility certificate, defined as

              || G'*z + A'*y || / max(1, ||c||).

        - 'residual as dual infeasibility certificate': None.
        If the default solver is used, the residual as primal infeasiblity
        certificate is guaranteed to be less than
        solvers.options['feastol'] (default 1e-7).  For the other
        solvers, the default GLPK or MOSEK exit criteria apply.

        Status 'dual infeasible'.  If the GLPK solver is used, all the
        fields except the status field are empty.  For the default and the
        MOSEK solvers, the values are as follows.
        - 'x', 's' are an approximate proof of dual infeasibility

              c'*x = -1,  G*x + s = 0,  A*x = 0,  s >= 0.

        - 'y', 'z': None.
        - 'primal objective': -1.0.
        - 'dual objective': None.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': the smallest primal slack min_k s_k .
        - 'dual slack': None.
        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate: the residual in
          the conditions of the infeasibility certificate, defined as
          the maximum of

              || G*x + s || / max(1, ||h||) and || A*x || / max(1, ||b||).

        If the default solver is used, the residual as dual infeasiblity
        certificate is guaranteed to be less than
        solvers.options['feastol'] (default 1e-7).  For the other
        solvers, the default GLPK or MOSEK exit criteria apply.

        Status 'unknown'.  If the GLPK or MOSEK solver is used, all the
        fields except the status field are None.  If the default solver
        is used, the values are as follows.
        - 'x', 'y', 's', 'z' are the last iterates before termination.
          These satisfy s > 0 and z > 0, but are not necessarily feasible.
        - 'primal objective': the primal cost c'*x.
        - 'dual objective': the dual cost -h'*z - b'*y.
        - 'gap': the duality gap s'*z.
        - 'relative gap': the relative gap, defined as s'*z / -c'*x if the
          primal cost is negative, s'*z / -(h'*z + b'*y) if the dual cost
          is positive, and None otherwise.
        - 'primal infeasibility ': the residual in the primal constraints,
          defined as the maximum of the residual in the inequalities

              || G*x + s - h || / max(1, ||h||)

          and the residual in the equalities

              || A*x - b || / max(1, ||b||).

        - 'dual infeasibility': the residual in the dual constraints,
          defined as

              || G'*z + A'*y + c || / max(1, ||c||).

        - 'primal slack': the smallest primal slack min_k s_k.
        - 'dual slack': the smallest dual slack min_k z_k.
        - 'residual as primal infeasibility certificate':
           None if h'*z + b'*y >= 0; the residual

              || G'*z + A'*y || / (-(h'*z + b'*y) * max(1, ||c||) )

          otherwise.
        - 'residual as dual infeasibility certificate':
          None if c'*x >= 0; the maximum of the residuals

              || G*x + s || / (-c'*x * max(1, ||h||))

          and

              || A*x || / (-c'*x * max(1, ||b||))

          otherwise.
        Termination with status 'unknown' indicates that the algorithm
        failed to find a solution that satisfies the specified tolerances.
        In some cases, the returned solution may be fairly accurate.  If
        the primal and dual infeasibilities, the gap, and the relative gap
        are small, then x, y, s, z are close to optimal.  If the residual
        as primal infeasibility certificate is small, then

            y / (-h'*z - b'*y),   z / (-h'*z - b'*y)

        provide an approximate certificate of primal infeasibility.  If
        the residual as certificate of dual infeasibility is small, then

            x / (-c'*x),   s / (-c'*x)

        provide an approximate proof of dual infeasibility.


    Control parameters.

        The control parameters for the different solvers can be modified
        by adding an entry to the dictionary cvxopt.solvers.options.  The
        following parameters control the execution of the default solver.

            options['show_progress'] True/False (default: True)
            options['maxiters'] positive integer (default: 100)
            options['refinement']  positive integer (default: 0)
            options['abstol'] scalar (default: 1e-7)
            options['reltol'] scalar (default: 1e-6)
            options['feastol'] scalar (default: 1e-7).

        The control parameter names for GLPK are strings with the name of
        the GLPK parameter, listed in the GLPK documentation.  The MOSEK
        parameters can me modified by adding an entry options['mosek'],
        containing a dictionary with MOSEK parameter/value pairs, as
        described in the MOSEK documentation.

        Options that are not recognized are replaced by their default
        values.
    """
    options = kwargs.get('options', globals()['options'])
    import math
    from cvxopt import base, blas, misc
    from cvxopt.base import matrix, spmatrix
    if not isinstance(c, matrix) or c.typecode != 'd' or c.size[1] != 1:
        raise TypeError("'c' must be a dense column matrix")
    n = c.size[0]
    if n < 1:
        raise ValueError('number of variables must be at least 1')
    if not isinstance(G, (matrix, spmatrix)) or G.typecode != 'd' or G.size[1] != n:
        raise TypeError("'G' must be a dense or sparse 'd' matrix with %d columns" % n)
    m = G.size[0]
    if not isinstance(h, matrix) or h.typecode != 'd' or h.size != (m, 1):
        raise TypeError("'h' must be a 'd' matrix of size (%d,1)" % m)
    if A is None:
        A = spmatrix([], [], [], (0, n), 'd')
    if not isinstance(A, (matrix, spmatrix)) or A.typecode != 'd' or A.size[1] != n:
        raise TypeError("'A' must be a dense or sparse 'd' matrix with %d columns" % n)
    p = A.size[0]
    if b is None:
        b = matrix(0.0, (0, 1))
    if not isinstance(b, matrix) or b.typecode != 'd' or b.size != (p, 1):
        raise TypeError("'b' must be a dense matrix of size (%d,1)" % p)
    if solver == 'glpk':
        try:
            from cvxopt import glpk
        except ImportError:
            raise ValueError("invalid option (solver = 'glpk'): cvxopt.glpk is not installed")
        opts = options.get('glpk', None)
        if opts:
            status, x, z, y = glpk.lp(c, G, h, A, b, options=opts)
        else:
            status, x, z, y = glpk.lp(c, G, h, A, b)
        if status == 'optimal':
            resx0 = max(1.0, blas.nrm2(c))
            resy0 = max(1.0, blas.nrm2(b))
            resz0 = max(1.0, blas.nrm2(h))
            pcost = blas.dot(c, x)
            dcost = -blas.dot(h, z) - blas.dot(b, y)
            s = matrix(h)
            base.gemv(G, x, s, alpha=-1.0, beta=1.0)
            gap = blas.dot(s, z)
            if pcost < 0.0:
                relgap = gap / -pcost
            elif dcost > 0.0:
                relgap = gap / dcost
            else:
                relgap = None
            rx = matrix(c)
            base.gemv(G, z, rx, beta=1.0, trans='T')
            base.gemv(A, y, rx, beta=1.0, trans='T')
            resx = blas.nrm2(rx) / resx0
            ry = matrix(b)
            base.gemv(A, x, ry, alpha=-1.0, beta=1.0)
            resy = blas.nrm2(ry) / resy0
            rz = matrix(0.0, (m, 1))
            base.gemv(G, x, rz)
            blas.axpy(s, rz)
            blas.axpy(h, rz, alpha=-1.0)
            resz = blas.nrm2(rz) / resz0
            dims = {'l': m, 's': [], 'q': []}
            pslack = -misc.max_step(s, dims)
            dslack = -misc.max_step(z, dims)
            pres, dres = (max(resy, resz), resx)
            pinfres, dinfres = (None, None)
        else:
            s = None
            pcost, dcost = (None, None)
            gap, relgap = (None, None)
            pres, dres = (None, None)
            pslack, dslack = (None, None)
            pinfres, dinfres = (None, None)
        return {'status': status, 'x': x, 's': s, 'y': y, 'z': z, 'primal objective': pcost, 'dual objective': dcost, 'gap': gap, 'relative gap': relgap, 'primal infeasibility': pres, 'dual infeasibility': dres, 'primal slack': pslack, 'dual slack': dslack, 'residual as primal infeasibility certificate': pinfres, 'residual as dual infeasibility certificate': dinfres}
    if solver == 'mosek':
        try:
            from cvxopt import msk
            import mosek
        except ImportError:
            raise ValueError("invalid option (solver = 'mosek'): cvxopt.msk is not installed")
        opts = options.get('mosek', None)
        if opts:
            solsta, x, z, y = msk.lp(c, G, h, A, b, options=opts)
        else:
            solsta, x, z, y = msk.lp(c, G, h, A, b)
        resx0 = max(1.0, blas.nrm2(c))
        resy0 = max(1.0, blas.nrm2(b))
        resz0 = max(1.0, blas.nrm2(h))
        if solsta in (mosek.solsta.optimal, getattr(mosek.solsta, 'near_optimal', None)):
            if solsta is mosek.solsta.optimal:
                status = 'optimal'
            else:
                status = 'near optimal'
            pcost = blas.dot(c, x)
            dcost = -blas.dot(h, z) - blas.dot(b, y)
            s = matrix(h)
            base.gemv(G, x, s, alpha=-1.0, beta=1.0)
            gap = blas.dot(s, z)
            if pcost < 0.0:
                relgap = gap / -pcost
            elif dcost > 0.0:
                relgap = gap / dcost
            else:
                relgap = None
            rx = matrix(c)
            base.gemv(G, z, rx, beta=1.0, trans='T')
            base.gemv(A, y, rx, beta=1.0, trans='T')
            resx = blas.nrm2(rx) / resx0
            ry = matrix(b)
            base.gemv(A, x, ry, alpha=-1.0, beta=1.0)
            resy = blas.nrm2(ry) / resy0
            rz = matrix(0.0, (m, 1))
            base.gemv(G, x, rz)
            blas.axpy(s, rz)
            blas.axpy(h, rz, alpha=-1.0)
            resz = blas.nrm2(rz) / resz0
            dims = {'l': m, 's': [], 'q': []}
            pslack = -misc.max_step(s, dims)
            dslack = -misc.max_step(z, dims)
            pres, dres = (max(resy, resz), resx)
            pinfres, dinfres = (None, None)
        elif solsta is mosek.solsta.prim_infeas_cer:
            status = 'primal infeasible'
            hz, by = (blas.dot(h, z), blas.dot(b, y))
            blas.scal(1.0 / (-hz - by), y)
            blas.scal(1.0 / (-hz - by), z)
            rx = matrix(0.0, (n, 1))
            base.gemv(A, y, rx, alpha=-1.0, trans='T')
            base.gemv(G, z, rx, alpha=-1.0, beta=1.0, trans='T')
            pinfres = blas.nrm2(rx) / resx0
            dinfres = None
            x, s = (None, None)
            pres, dres = (None, None)
            pcost, dcost = (None, 1.0)
            gap, relgap = (None, None)
            dims = {'l': m, 's': [], 'q': []}
            dslack = -misc.max_step(z, dims)
            pslack = None
        elif solsta == mosek.solsta.dual_infeas_cer:
            status = 'dual infeasible'
            cx = blas.dot(c, x)
            blas.scal(-1.0 / cx, x)
            s = matrix(0.0, (m, 1))
            base.gemv(G, x, s, alpha=-1.0)
            ry = matrix(0.0, (p, 1))
            base.gemv(A, x, ry)
            resy = blas.nrm2(ry) / resy0
            rz = matrix(s)
            base.gemv(G, x, rz, beta=1.0)
            resz = blas.nrm2(rz) / resz0
            pres, dres = (None, None)
            dinfres, pinfres = (max(resy, resz), None)
            z, y = (None, None)
            pcost, dcost = (-1.0, None)
            gap, relgap = (None, None)
            dims = {'l': m, 's': [], 'q': []}
            pslack = -misc.max_step(s, dims)
            dslack = None
        else:
            status = 'unknown'
            s = None
            pcost, dcost = (None, None)
            gap, relgap = (None, None)
            pres, dres = (None, None)
            pinfres, dinfres = (None, None)
            pslack, dslack = (None, None)
        return {'status': status, 'x': x, 's': s, 'y': y, 'z': z, 'primal objective': pcost, 'dual objective': dcost, 'gap': gap, 'relative gap': relgap, 'primal infeasibility': pres, 'dual infeasibility': dres, 'residual as primal infeasibility certificate': pinfres, 'residual as dual infeasibility certificate': dinfres, 'primal slack': pslack, 'dual slack': dslack}
    return conelp(c, G, h, {'l': m, 'q': [], 's': []}, A, b, primalstart, dualstart, kktsolver=kktsolver, options=options)
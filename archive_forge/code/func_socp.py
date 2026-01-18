import sys
def socp(c, Gl=None, hl=None, Gq=None, hq=None, A=None, b=None, kktsolver=None, solver=None, primalstart=None, dualstart=None, **kwargs):
    """
    Solves a pair of primal and dual SOCPs

        minimize    c'*x
        subject to  Gl*x + sl = hl
                    Gq[k]*x + sq[k] = hq[k],  k = 0, ..., N-1
                    A*x = b
                    sl >= 0,
                    sq[k] >= 0, k = 0, ..., N-1

        maximize    -hl'*z - sum_k hq[k]'*zq[k] - b'*y
        subject to  Gl'*zl + sum_k Gq[k]'*zq[k] + A'*y + c = 0
                    zl >= 0,  zq[k] >= 0, k = 0, ..., N-1.

    The inequalities sl >= 0 and zl >= 0 are elementwise vector
    inequalities.  The inequalities sq[k] >= 0, zq[k] >= 0 are second
    order cone inequalities, i.e., equivalent to

        sq[k][0] >= || sq[k][1:] ||_2,  zq[k][0] >= || zq[k][1:] ||_2.


    Input arguments.

        Gl is a dense or sparse 'd' matrix of size (ml, n).  hl is a
        dense 'd' matrix of size (ml, 1). The default values of Gl and hl
        are matrices with zero rows.

        The argument Gq is a list of N dense or sparse 'd' matrices of
        size (m[k] n), k = 0, ..., N-1, where m[k] >= 1.  hq is a list
        of N dense 'd' matrices of size (m[k], 1), k = 0, ..., N-1.
        The default values of Gq and hq are empty lists.

        A is a dense or sparse 'd' matrix of size (p,1).  b is a dense 'd'
        matrix of size (p,1).  The default values of A and b are matrices
        with zero rows.

        solver is None or 'mosek'.  The default solver (None) uses the
        cvxopt conelp() function.  The 'mosek' solver is the SOCP solver
        from MOSEK.

        The arguments primalstart and dualstart are ignored when solver
        is 'mosek', and are optional when solver is None.

        The argument primalstart is a dictionary with keys 'x', 'sl', 'sq',
        and specifies an optional primal starting point.
        primalstart['x'] is a dense 'd' matrix of size (n,1).
        primalstart['sl'] is a positive dense 'd' matrix of size (ml,1).
        primalstart['sq'] is a list of matrices of size (m[k],1), positive
        with respect to the second order cone of order m[k].

        The argument dualstart is a dictionary with keys 'y', 'zl', 'zq',
        and specifies an optional dual starting point.
        dualstart['y'] is a dense 'd' matrix of size (p,1).
        dualstart['zl'] is a positive dense 'd' matrix of size (ml,1).
        dualstart['sq'] is a list of matrices of size (m[k],1), positive
        with respect to the second order cone of order m[k].


    Output arguments.

        Returns a dictionary with keys 'status', 'x', 'sl', 'sq', 'zl',
        'zq', 'y', 'primal objective', 'dual objective', 'gap',
        'relative gap',  'primal infeasibility', 'dual infeasibility',
        'primal slack', 'dual slack', 'residual as primal infeasibility
        certificate', 'residual as dual infeasibility certificate'.

        The 'status' field has values 'optimal', 'primal infeasible',
        'dual infeasible', or 'unknown'.  The values of the other fields
        depend on the exit status and the solver used.

        Status 'optimal'.
        - 'x', 'sl', 'sq', 'y', 'zl', 'zq' are an approximate solution of
          the primal and dual optimality conditions

              G*x + s = h,  A*x = b
              G'*z + A'*y + c = 0
              s >= 0, z >= 0
              s'*z = 0

          where

              G = [ Gl; Gq[0]; ...; Gq[N-1] ]
              h = [ hl; hq[0]; ...; hq[N-1] ]
              s = [ sl; sq[0]; ...; sq[N-1] ]
              z = [ zl; zq[0]; ...; zq[N-1] ].

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

        - 'primal slack': the smallest primal slack,

              min( min_k sl_k, min_k (sq[k][0] - || sq[k][1:] ||) ).

        - 'dual slack': the smallest dual slack,

              min( min_k zl_k, min_k (zq[k][0] - || zq[k][1:] ||) ).

        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate': None.
        If the default solver is used, the primal infeasibility is
        guaranteed to be less than solvers.options['feastol']
        (default 1e-7).  The dual infeasibility is guaranteed to be less
        than solvers.options['feastol'] (default 1e-7).  The gap is less
        than solvers.options['abstol'] (default 1e-7) or the relative gap
        is less than solvers.options['reltol'] (default 1e-6).
        If the MOSEK solver is used, the default MOSEK exit criteria
        apply.

        Status 'primal infeasible'.
        - 'x', 'sl', 'sq': None.
        - 'y', 'zl', 'zq' are an approximate certificate of infeasibility

              -h'*z - b'*y = 1,  G'*z + A'*y = 0,  z >= 0.

        - 'primal objective': None.
        - 'dual objective': 1.0.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': None.
        - 'dual slack': the smallest dual slack,

              min( min_k zl_k, min_k (zq[k][0] - || zq[k][1:] ||) ).

        - 'residual as primal infeasibility certificate': the residual in
          the condition of the infeasibility certificate, defined as

              || G'*z + A'*y || / max(1, ||c||).

        - 'residual as dual infeasibility certificate': None.
        If the default solver is used, the residual as primal infeasiblity
        certificate is guaranteed to be less than
        solvers.options['feastol'] (default 1e-7).  If the MOSEK solver is
        used, the default MOSEK exit criteria apply.

        Status 'dual infeasible'.
        - 'x', 'sl', 'sq': an approximate proof of dual infeasibility

              c'*x = -1,  G*x + s = 0,  A*x = 0,  s >= 0.

        - 'y', 'zl', 'zq': None.
        - 'primal objective': -1.0.
        - 'dual objective': None.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': the smallest primal slack,

              min( min_k sl_k, min_k (sq[k][0] - || sq[k][1:] ||) ).

        - 'dual slack': None.
        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate: the residual in
          the conditions of the infeasibility certificate, defined as
          the maximum of

              || G*x + s || / max(1, ||h||) and || A*x || / max(1, ||b||).

        If the default solver is used, the residual as dual infeasiblity
        certificate is guaranteed to be less than
        solvers.options['feastol'] (default 1e-7).  If the MOSEK solver
        is used, the default MOSEK exit criteria apply.

        Status 'unknown'.  If the MOSEK solver is used, all the fields
        except the status field are empty.  If the default solver
        is used, the values are as follows.
        - 'x', 'y', 'sl', 'sq', 'zl', 'zq': the last iterates before
          termination.   These satisfy s > 0 and z > 0, but are not
          necessarily feasible.
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

        - 'primal slack': the smallest primal slack,

              min( min_k sl_k, min_k (sq[k][0] - || sq[k][1:] ||) ).

        - 'dual slack': the smallest dual slack,

              min( min_k zl_k, min_k (zq[k][0] - || zq[k][1:] ||) ).

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
            options['refinement'] positive integer (default: 1)
            options['abstol'] scalar (default: 1e-7)
            options['reltol'] scalar (default: 1e-6)
            options['feastol'] scalar (default: 1e-7).

        The MOSEK parameters can me modified by adding an entry
        options['mosek'], containing a dictionary with MOSEK
        parameter/value pairs, as described in the MOSEK documentation.

        Options that are not recognized are replaced by their default
        values.
    """
    from cvxopt import base, blas
    from cvxopt.base import matrix, spmatrix
    if not isinstance(c, matrix) or c.typecode != 'd' or c.size[1] != 1:
        raise TypeError("'c' must be a dense column matrix")
    n = c.size[0]
    if n < 1:
        raise ValueError('number of variables must be at least 1')
    if Gl is None:
        Gl = spmatrix([], [], [], (0, n), tc='d')
    if not isinstance(Gl, (matrix, spmatrix)) or Gl.typecode != 'd' or Gl.size[1] != n:
        raise TypeError("'Gl' must be a dense or sparse 'd' matrix with %d columns" % n)
    ml = Gl.size[0]
    if hl is None:
        hl = matrix(0.0, (0, 1))
    if not isinstance(hl, matrix) or hl.typecode != 'd' or hl.size != (ml, 1):
        raise TypeError("'hl' must be a dense 'd' matrix of size (%d,1)" % ml)
    if Gq is None:
        Gq = []
    if not isinstance(Gq, list) or [G for G in Gq if not isinstance(G, (matrix, spmatrix)) or G.typecode != 'd' or G.size[1] != n]:
        raise TypeError("'Gq' must be a list of sparse or dense 'd' matrices with %d columns" % n)
    mq = [G.size[0] for G in Gq]
    a = [k for k in range(len(mq)) if mq[k] == 0]
    if a:
        raise TypeError('the number of rows of Gq[%d] is zero' % a[0])
    if hq is None:
        hq = []
    if not isinstance(hq, list) or len(hq) != len(mq) or [h for h in hq if not isinstance(h, (matrix, spmatrix)) or h.typecode != 'd']:
        raise TypeError("'hq' must be a list of %d dense or sparse 'd' matrices" % len(mq))
    a = [k for k in range(len(mq)) if hq[k].size != (mq[k], 1)]
    if a:
        k = a[0]
        raise TypeError("'hq[%d]' has size (%d,%d).  Expected size is (%d,1)." % (k, hq[k].size[0], hq[k].size[1], mq[k]))
    if A is None:
        A = spmatrix([], [], [], (0, n), 'd')
    if not isinstance(A, (matrix, spmatrix)) or A.typecode != 'd' or A.size[1] != n:
        raise TypeError("'A' must be a dense or sparse 'd' matrix with %d columns" % n)
    p = A.size[0]
    if b is None:
        b = matrix(0.0, (0, 1))
    if not isinstance(b, matrix) or b.typecode != 'd' or b.size != (p, 1):
        raise TypeError("'b' must be a dense matrix of size (%d,1)" % p)
    dims = {'l': ml, 'q': mq, 's': []}
    N = ml + sum(mq)
    if solver == 'mosek':
        from cvxopt import misc
        try:
            from cvxopt import msk
            import mosek
        except ImportError:
            raise ValueError("invalid option (solver = 'mosek'): cvxopt.msk is not installed")
        if p:
            raise ValueError("socp() with the solver = 'mosek' option does not handle problems with equality constraints")
        opts = options.get('mosek', None)
        if opts:
            solsta, x, zl, zq = msk.socp(c, Gl, hl, Gq, hq, options=opts)
        else:
            solsta, x, zl, zq = msk.socp(c, Gl, hl, Gq, hq)
        resx0 = max(1.0, blas.nrm2(c))
        rh = matrix([blas.nrm2(hl)] + [blas.nrm2(hqk) for hqk in hq])
        resz0 = max(1.0, blas.nrm2(rh))
        if solsta in (mosek.solsta.optimal, getattr(mosek.solsta, 'near_optimal')):
            if solsta is mosek.solsta.optimal:
                status = 'optimal'
            else:
                status = 'near optimal'
            y = matrix(0.0, (0, 1))
            pcost = blas.dot(c, x)
            dcost = -blas.dot(hl, zl) - sum([blas.dot(hq[k], zq[k]) for k in range(len(mq))])
            sl = matrix(hl)
            base.gemv(Gl, x, sl, alpha=-1.0, beta=1.0)
            sq = [+hqk for hqk in hq]
            for k in range(len(Gq)):
                base.gemv(Gq[k], x, sq[k], alpha=-1.0, beta=1.0)
            gap = blas.dot(sl, zl) + sum([blas.dot(zq[k], sq[k]) for k in range(len(mq))])
            if pcost < 0.0:
                relgap = gap / -pcost
            elif dcost > 0.0:
                relgap = gap / dcost
            else:
                relgap = None
            rx = matrix(c)
            base.gemv(Gl, zl, rx, beta=1.0, trans='T')
            for k in range(len(mq)):
                base.gemv(Gq[k], zq[k], rx, beta=1.0, trans='T')
            resx = blas.nrm2(rx) / resx0
            rz = matrix(0.0, (ml + sum(mq), 1))
            base.gemv(Gl, x, rz)
            blas.axpy(sl, rz)
            blas.axpy(hl, rz, alpha=-1.0)
            ind = ml
            for k in range(len(mq)):
                base.gemv(Gq[k], x, rz, offsety=ind)
                blas.axpy(sq[k], rz, offsety=ind)
                blas.axpy(hq[k], rz, alpha=-1.0, offsety=ind)
                ind += mq[k]
            resz = blas.nrm2(rz) / resz0
            s, z = (matrix(0.0, (N, 1)), matrix(0.0, (N, 1)))
            blas.copy(sl, s)
            blas.copy(zl, z)
            ind = ml
            for k in range(len(mq)):
                blas.copy(zq[k], z, offsety=ind)
                blas.copy(sq[k], s, offsety=ind)
                ind += mq[k]
            pslack = -misc.max_step(s, dims)
            dslack = -misc.max_step(z, dims)
            pres, dres = (resz, resx)
            pinfres, dinfres = (None, None)
        elif solsta is mosek.solsta.dual_infeas_cer:
            status = 'primal infeasible'
            y = matrix(0.0, (0, 1))
            hz = blas.dot(hl, zl) + sum([blas.dot(hq[k], zq[k]) for k in range(len(mq))])
            blas.scal(1.0 / -hz, zl)
            for k in range(len(mq)):
                blas.scal(1.0 / -hz, zq[k])
            x, sl, sq = (None, None, None)
            rx = matrix(0.0, (n, 1))
            base.gemv(Gl, zl, rx, alpha=-1.0, beta=1.0, trans='T')
            for k in range(len(mq)):
                base.gemv(Gq[k], zq[k], rx, beta=1.0, trans='T')
            pinfres = blas.nrm2(rx) / resx0
            dinfres = None
            z = matrix(0.0, (N, 1))
            blas.copy(zl, z)
            ind = ml
            for k in range(len(mq)):
                blas.copy(zq[k], z, offsety=ind)
                ind += mq[k]
            dslack = -misc.max_step(z, dims)
            pslack = None
            x, s = (None, None)
            pres, dres = (None, None)
            pcost, dcost = (None, 1.0)
            gap, relgap = (None, None)
        elif solsta == mosek.solsta.prim_infeas_cer:
            status = 'dual infeasible'
            cx = blas.dot(c, x)
            blas.scal(-1.0 / cx, x)
            sl = matrix(0.0, (ml, 1))
            base.gemv(Gl, x, sl, alpha=-1.0)
            sq = [matrix(0.0, (mqk, 1)) for mqk in mq]
            for k in range(len(mq)):
                base.gemv(Gq[k], x, sq[k], alpha=-1.0, beta=1.0)
            rz = matrix([sl] + [sqk for sqk in sq])
            base.gemv(Gl, x, rz, beta=1.0)
            ind = ml
            for k in range(len(mq)):
                base.gemv(Gq[k], x, rz, beta=1.0, offsety=ind)
                ind += mq[k]
            resz = blas.nrm2(rz) / resz0
            dims = {'l': ml, 's': [], 'q': mq}
            s = matrix(0.0, (N, 1))
            blas.copy(sl, s)
            ind = ml
            for k in range(len(mq)):
                blas.copy(sq[k], s, offsety=ind)
                ind += mq[k]
            pslack = -misc.max_step(s, dims)
            dslack = None
            pres, dres = (None, None)
            dinfres, pinfres = (resz, None)
            z, y = (None, None)
            pcost, dcost = (-1.0, None)
            gap, relgap = (None, None)
        else:
            status = 'unknown'
            sl, sq = (None, None)
            zl, zq = (None, None)
            x, y = (None, None)
            pcost, dcost = (None, None)
            gap, relgap = (None, None)
            pres, dres = (None, None)
            pinfres, dinfres = (None, None)
            pslack, dslack = (None, None)
        return {'status': status, 'x': x, 'sl': sl, 'sq': sq, 'y': y, 'zl': zl, 'zq': zq, 'primal objective': pcost, 'dual objective': dcost, 'gap': gap, 'relative gap': relgap, 'primal infeasibility': pres, 'dual infeasibility': dres, 'residual as primal infeasibility certificate': pinfres, 'residual as dual infeasibility certificate': dinfres, 'primal slack': pslack, 'dual slack': dslack}
    h = matrix(0.0, (N, 1))
    if isinstance(Gl, matrix) or [Gk for Gk in Gq if isinstance(Gk, matrix)]:
        G = matrix(0.0, (N, n))
    else:
        G = spmatrix([], [], [], (N, n), 'd')
    h[:ml] = hl
    G[:ml, :] = Gl
    ind = ml
    for k in range(len(mq)):
        h[ind:ind + mq[k]] = hq[k]
        G[ind:ind + mq[k], :] = Gq[k]
        ind += mq[k]
    if primalstart:
        ps = {}
        ps['x'] = primalstart['x']
        ps['s'] = matrix(0.0, (N, 1))
        if ml:
            ps['s'][:ml] = primalstart['sl']
        if mq:
            ind = ml
            for k in range(len(mq)):
                ps['s'][ind:ind + mq[k]] = primalstart['sq'][k][:]
                ind += mq[k]
    else:
        ps = None
    if dualstart:
        ds = {}
        if p:
            ds['y'] = dualstart['y']
        ds['z'] = matrix(0.0, (N, 1))
        if ml:
            ds['z'][:ml] = dualstart['zl']
        if mq:
            ind = ml
            for k in range(len(mq)):
                ds['z'][ind:ind + mq[k]] = dualstart['zq'][k][:]
                ind += mq[k]
    else:
        ds = None
    sol = conelp(c, G, h, dims, A=A, b=b, primalstart=ps, dualstart=ds, kktsolver=kktsolver, options=options)
    if sol['s'] is None:
        sol['sl'] = None
        sol['sq'] = None
    else:
        sol['sl'] = sol['s'][:ml]
        sol['sq'] = [matrix(0.0, (m, 1)) for m in mq]
        ind = ml
        for k in range(len(mq)):
            sol['sq'][k][:] = sol['s'][ind:ind + mq[k]]
            ind += mq[k]
    del sol['s']
    if sol['z'] is None:
        sol['zl'] = None
        sol['zq'] = None
    else:
        sol['zl'] = sol['z'][:ml]
        sol['zq'] = [matrix(0.0, (m, 1)) for m in mq]
        ind = ml
        for k in range(len(mq)):
            sol['zq'][k][:] = sol['z'][ind:ind + mq[k]]
            ind += mq[k]
    del sol['z']
    return sol
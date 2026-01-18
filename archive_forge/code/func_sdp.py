import sys
def sdp(c, Gl=None, hl=None, Gs=None, hs=None, A=None, b=None, kktsolver=None, solver=None, primalstart=None, dualstart=None, **kwargs):
    """

    Solves a pair of primal and dual SDPs

        minimize    c'*x
        subject to  Gl*x + sl = hl
                    mat(Gs[k]*x) + ss[k] = hs[k], k = 0, ..., N-1
                    A*x = b
                    sl >= 0,  ss[k] >= 0, k = 0, ..., N-1

        maximize    -hl'*z - sum_k trace(hs[k]*zs[k]) - b'*y
        subject to  Gl'*zl + sum_k Gs[k]'*vec(zs[k]) + A'*y + c = 0
                    zl >= 0,  zs[k] >= 0, k = 0, ..., N-1.

    The inequalities sl >= 0 and zl >= 0 are elementwise vector
    inequalities.  The inequalities ss[k] >= 0, zs[k] >= 0 are matrix
    inequalities, i.e., the symmetric matrices ss[k] and zs[k] must be
    positive semidefinite.  mat(Gs[k]*x) is the symmetric matrix X with
    X[:] = Gs[k]*x.  For a symmetric matrix, zs[k], vec(zs[k]) is the
    vector zs[k][:].


    Input arguments.

        Gl is a dense or sparse 'd' matrix of size (ml, n).  hl is a
        dense 'd' matrix of size (ml, 1). The default values of Gl and hl
        are matrices with zero rows.

        The argument Gs is a list of N dense or sparse 'd' matrices of
        size (m[k]**2, n), k = 0, ..., N-1.   The columns of Gs[k]
        represent symmetric matrices stored as vectors in column major
        order.  hs is a list of N dense 'd' matrices of size (m[k], m[k]),
        k = 0, ..., N-1.  The columns of Gs[k] and the matrices hs[k]
        represent symmetric matrices in 'L' storage, i.e., only the lower
        triangular elements are accessed.  The default values of Gs and
        hs are empty lists.

        A is a dense or sparse 'd' matrix of size (p,n).  b is a dense 'd'
        matrix of size (p,1).  The default values of A and b are matrices
        with zero rows.

        solver is None or 'dsdp'.  The default solver (None) calls
        cvxopt.conelp().  The 'dsdp' solver uses an interface to DSDP5.
        The 'dsdp' solver does not accept problems with equality
        constraints (A and b must have zero rows, or be absent).

        The argument primalstart is a dictionary with keys 'x', 'sl',
        'ss', and specifies an optional primal starting point.
        primalstart['x'] is a dense 'd' matrix of length n;
        primalstart['sl'] is a  positive dense 'd' matrix of length ml;
        primalstart['ss'] is a list of positive definite matrices of
        size (ms[k], ms[k]).  Only the lower triangular parts of these
        matrices will be accessed.

        The argument dualstart is a dictionary with keys 'zl', 'zs', 'y'
        and specifies an optional dual starting point.
        dualstart['y'] is a dense 'd' matrix of length p;
        dualstart['zl'] must be a positive dense 'd' matrix of length ml;
        dualstart['zs'] is a list of positive definite matrices of
        size (ms[k], ms[k]).  Only the lower triangular parts of these
        matrices will be accessed.

        The arguments primalstart and dualstart are ignored when solver
        is 'dsdp'.


    Output arguments.

        Returns a dictionary with keys 'status', 'x', 'sl', 'ss', 'zl',
        'zs', 'y', 'primal objective', 'dual objective', 'gap',
        'relative gap',  'primal infeasibility', 'dual infeasibility',
        'primal slack', 'dual slack', 'residual as primal infeasibility
        certificate', 'residual as dual infeasibility certificate'.

        The 'status' field has values 'optimal', 'primal infeasible',
        'dual infeasible', or 'unknown'.  The values of the other fields
        depend on the exit status and the solver used.

        Status 'optimal'.
        - 'x', 'sl', 'ss', 'y', 'zl', 'zs' are an approximate solution of
          the primal and dual optimality conditions

              G*x + s = h,  A*x = b
              G'*z + A'*y + c = 0
              s >= 0, z >= 0
              s'*z = 0

          where

              G = [ Gl; Gs[0][:]; ...; Gs[N-1][:] ]
              h = [ hl; hs[0][:]; ...; hs[N-1][:] ]
              s = [ sl; ss[0][:]; ...; ss[N-1][:] ]
              z = [ zl; zs[0][:]; ...; zs[N-1][:] ].

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

              min( min_k sl_k, min_k lambda_min(mat(ss[k])) ).

        - 'dual slack': the smallest dual slack,

              min( min_k zl_k, min_k lambda_min(mat(zs[k])) ).

        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate': None.
        If the default solver is used, the primal infeasibility is
        guaranteed to be less than solvers.options['feastol']
        (default 1e-7).  The dual infeasibility is guaranteed to be less
        than solvers.options['feastol'] (default 1e-7).  The gap is less
        than solvers.options['abstol'] (default 1e-7) or the relative gap
        is less than solvers.options['reltol'] (default 1e-6).
        If the DSDP solver is used, the default DSDP exit criteria
        apply.

        Status 'primal infeasible'.
        - 'x', 'sl', 'ss': None.
        - 'y', 'zl', 'zs' are an approximate certificate of infeasibility

              -h'*z - b'*y = 1,  G'*z + A'*y = 0,  z >= 0.

        - 'primal objective': None.
        - 'dual objective': 1.0.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': None
        - 'dual slack': the smallest dual slack,

              min( min_k zl_k, min_k lambda_min(mat(zs[k])) ).

        - 'residual as primal infeasibility certificate': the residual in
          the condition of the infeasibility certificate, defined as

              || G'*z + A'*y || / max(1, ||c||).

        - 'residual as dual infeasibility certificate': None.
        If the default solver is used, the residual as primal infeasiblity
        certificate is guaranteed to be less than
        solvers.options['feastol'] (default 1e-7).  If the DSDP solver is
        used, the default DSDP exit criteria apply.

        Status 'dual infeasible'.
        - 'x', 'sl', 'ss': an approximate proof of dual infeasibility

              c'*x = -1,  G*x + s = 0,  A*x = 0,  s >= 0.

        - 'y', 'zl', 'zs': None.
        - 'primal objective': -1.0.
        - 'dual objective': None.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': the smallest primal slack,

              min( min_k sl_k, min_k lambda_min(mat(ss[k])) ).

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

        Status 'unknown'.  If the DSDP solver is used, all the fields
        except the status field are empty.  If the default solver
        is used, the values are as follows.
        - 'x', 'y', 'sl', 'ss', 'zl', 'zs': the last iterates before
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

              min( min_k sl_k, min_k lambda_min(mat(ss[k])) ).

        - 'dual slack': the smallest dual slack,

              min( min_k zl_k, min_k lambda_min(mat(zs[k])) ).

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

        The following parameters control the execution of the default
        solver.

            options['show_progress'] True/False (default: True)
            options['maxiters'] positive integer (default: 100)
            options['refinement'] positive integer (default: 1)
            options['abstol'] scalar (default: 1e-7)
            options['reltol'] scalar (default: 1e-6)
            options['feastol'] scalar (default: 1e-7).

        The execution of the 'dsdp' solver is controlled by:

            options['DSDP_Monitor'] integer (default: 0)
            options['DSDP_MaxIts'] positive integer
            options['DSDP_GapTolerance'] scalar (default: 1e-5).
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
    if Gl is None:
        Gl = spmatrix([], [], [], (0, n), tc='d')
    if not isinstance(Gl, (matrix, spmatrix)) or Gl.typecode != 'd' or Gl.size[1] != n:
        raise TypeError("'Gl' must be a dense or sparse 'd' matrix with %d columns" % n)
    ml = Gl.size[0]
    if hl is None:
        hl = matrix(0.0, (0, 1))
    if not isinstance(hl, matrix) or hl.typecode != 'd' or hl.size != (ml, 1):
        raise TypeError("'hl' must be a 'd' matrix of size (%d,1)" % ml)
    if Gs is None:
        Gs = []
    if not isinstance(Gs, list) or [G for G in Gs if not isinstance(G, (matrix, spmatrix)) or G.typecode != 'd' or G.size[1] != n]:
        raise TypeError("'Gs' must be a list of sparse or dense 'd' matrices with %d columns" % n)
    ms = [int(math.sqrt(G.size[0])) for G in Gs]
    a = [k for k in range(len(ms)) if ms[k] ** 2 != Gs[k].size[0]]
    if a:
        raise TypeError("the squareroot of the number of rows in 'Gs[%d]' is not an integer" % k)
    if hs is None:
        hs = []
    if not isinstance(hs, list) or len(hs) != len(ms) or [h for h in hs if not isinstance(h, (matrix, spmatrix)) or h.typecode != 'd']:
        raise TypeError("'hs' must be a list of %d dense or sparse 'd' matrices" % len(ms))
    a = [k for k in range(len(ms)) if hs[k].size != (ms[k], ms[k])]
    if a:
        k = a[0]
        raise TypeError('hs[%d] has size (%d,%d).  Expected size is (%d,%d).' % (k, hs[k].size[0], hs[k].size[1], ms[k], ms[k]))
    if A is None:
        A = spmatrix([], [], [], (0, n), 'd')
    if not isinstance(A, (matrix, spmatrix)) or A.typecode != 'd' or A.size[1] != n:
        raise TypeError("'A' must be a dense or sparse 'd' matrix with %d columns" % n)
    p = A.size[0]
    if b is None:
        b = matrix(0.0, (0, 1))
    if not isinstance(b, matrix) or b.typecode != 'd' or b.size != (p, 1):
        raise TypeError("'b' must be a dense matrix of size (%d,1)" % p)
    dims = {'l': ml, 'q': [], 's': ms}
    N = ml + sum([m ** 2 for m in ms])
    if solver == 'dsdp':
        try:
            from cvxopt import dsdp
        except ImportError:
            raise ValueError("invalid option (solver = 'dsdp'): cvxopt.dsdp is not installed")
        if p:
            raise ValueError("sdp() with the solver = 'dsdp' option does not handle problems with equality constraints")
        opts = options.get('dsdp', None)
        if opts:
            dsdpstatus, x, r, zl, zs = dsdp.sdp(c, Gl, hl, Gs, hs, options=opts)
        else:
            dsdpstatus, x, r, zl, zs = dsdp.sdp(c, Gl, hl, Gs, hs)
        resx0 = max(1.0, blas.nrm2(c))
        rh = matrix([blas.nrm2(hl)] + [math.sqrt(misc.sdot2(hsk, hsk)) for hsk in hs])
        resz0 = max(1.0, blas.nrm2(rh))
        if dsdpstatus == 'DSDP_UNBOUNDED':
            status = 'dual infeasible'
            cx = blas.dot(c, x)
            blas.scal(-1.0 / cx, x)
            sl = -Gl * x
            ss = [-matrix(Gs[k] * x, (ms[k], ms[k])) for k in range(len(ms))]
            for k in range(len(ms)):
                misc.symm(ss[k], ms[k])
            rz = matrix([sl] + [ssk[:] for ssk in ss])
            base.gemv(Gl, x, rz, beta=1.0)
            ind = ml
            for k in range(len(ms)):
                base.gemv(Gs[k], x, rz, beta=1.0, offsety=ind)
                ind += ms[k] ** 2
            dims = {'l': ml, 's': ms, 'q': []}
            resz = misc.nrm2(rz, dims) / resz0
            s = matrix(0.0, (N, 1))
            blas.copy(sl, s)
            ind = ml
            for k in range(len(ms)):
                blas.copy(ss[k], s, offsety=ind)
                ind += ms[k]
            pslack = -misc.max_step(s, dims)
            sslack = None
            pres, dres = (None, None)
            dinfres, pinfres = (resz, None)
            zl, zs, y = (None, None, None)
            pcost, dcost = (-1.0, None)
            gap, relgap = (None, None)
        elif dsdpstatus == 'DSDP_INFEASIBLE':
            status = 'primal infeasible'
            y = matrix(0.0, (0, 1))
            hz = blas.dot(hl, zl) + misc.sdot2(hs, zs)
            blas.scal(1.0 / -hz, zl)
            for k in range(len(ms)):
                blas.scal(1.0 / -hz, zs[k])
                misc.symm(zs[k], ms[k])
            rx = matrix(0.0, (n, 1))
            base.gemv(Gl, zl, rx, alpha=-1.0, beta=1.0, trans='T')
            ind = 0
            for k in range(len(ms)):
                blas.scal(0.5, zs[k], inc=ms[k] + 1)
                for j in range(ms[k]):
                    blas.scal(0.0, zs[k], offset=j + ms[k] * (j + 1), inc=ms[k])
                base.gemv(Gs[k], zs[k], rx, alpha=2.0, beta=1.0, trans='T')
                blas.scal(2.0, zs[k], inc=ms[k] + 1)
                ind += ms[k]
            pinfres = blas.nrm2(rx) / resx0
            dinfres = None
            z = matrix(0.0, (N, 1))
            blas.copy(zl, z)
            ind = ml
            for k in range(len(ms)):
                blas.copy(zs[k], z, offsety=ind)
                ind += ms[k]
            dslack = -misc.max_step(z, dims)
            pslack = None
            x, sl, ss = (None, None, None)
            pres, dres = (None, None)
            pcost, dcost = (None, 1.0)
            gap, relgap = (None, None)
        else:
            if dsdpstatus == 'DSDP_PDFEASIBLE':
                status = 'optimal'
            else:
                status = 'unknown'
            y = matrix(0.0, (0, 1))
            sl = hl - Gl * x
            ss = [hs[k] - matrix(Gs[k] * x, (ms[k], ms[k])) for k in range(len(ms))]
            for k in range(len(ms)):
                misc.symm(ss[k], ms[k])
                misc.symm(zs[k], ms[k])
            pcost = blas.dot(c, x)
            dcost = -blas.dot(hl, zl) - misc.sdot2(hs, zs)
            gap = blas.dot(sl, zl) + misc.sdot2(ss, zs)
            if pcost < 0.0:
                relgap = gap / -pcost
            elif dcost > 0.0:
                relgap = gap / dcost
            else:
                relgap = None
            rx = matrix(c)
            base.gemv(Gl, zl, rx, beta=1.0, trans='T')
            ind = 0
            for k in range(len(ms)):
                blas.scal(0.5, zs[k], inc=ms[k] + 1)
                for j in range(ms[k]):
                    blas.scal(0.0, zs[k], offset=j + ms[k] * (j + 1), inc=ms[k])
                base.gemv(Gs[k], zs[k], rx, alpha=2.0, beta=1.0, trans='T')
                blas.scal(2.0, zs[k], inc=ms[k] + 1)
                ind += ms[k]
            resx = blas.nrm2(rx) / resx0
            rz = matrix(0.0, (ml + sum([msk ** 2 for msk in ms]), 1))
            base.gemv(Gl, x, rz)
            blas.axpy(sl, rz)
            blas.axpy(hl, rz, alpha=-1.0)
            ind = ml
            for k in range(len(ms)):
                base.gemv(Gs[k], x, rz, offsety=ind)
                blas.axpy(ss[k], rz, offsety=ind, n=ms[k] ** 2)
                blas.axpy(hs[k], rz, alpha=-1.0, offsety=ind, n=ms[k] ** 2)
                ind += ms[k] ** 2
            resz = misc.snrm2(rz, dims) / resz0
            pres, dres = (resz, resx)
            s, z = (matrix(0.0, (N, 1)), matrix(0.0, (N, 1)))
            blas.copy(sl, s)
            blas.copy(zl, z)
            ind = ml
            for k in range(len(ms)):
                blas.copy(ss[k], s, offsety=ind)
                blas.copy(zs[k], z, offsety=ind)
                ind += ms[k]
            pslack = -misc.max_step(s, dims)
            dslack = -misc.max_step(z, dims)
            if status == 'optimal' or dcost <= 0.0:
                pinfres = None
            else:
                rx = matrix(0.0, (n, 1))
                base.gemv(Gl, zl, rx, beta=1.0, trans='T')
                ind = 0
                for k in range(len(ms)):
                    blas.scal(0.5, zs[k], inc=ms[k] + 1)
                    for j in range(ms[k]):
                        blas.scal(0.0, zs[k], offset=j + ms[k] * (j + 1), inc=ms[k])
                    base.gemv(Gs[k], zs[k], rx, alpha=2.0, beta=1.0, trans='T')
                    blas.scal(2.0, zs[k], inc=ms[k] + 1)
                    ind += ms[k]
                pinfres = blas.nrm2(rx) / resx0 / dcost
            if status == 'optimal' or pcost >= 0.0:
                dinfres = None
            else:
                rz = matrix(0.0, (ml + sum([msk ** 2 for msk in ms]), 1))
                base.gemv(Gl, x, rz)
                blas.axpy(sl, rz)
                ind = ml
                for k in range(len(ms)):
                    base.gemv(Gs[k], x, rz, offsety=ind)
                    blas.axpy(ss[k], rz, offsety=ind, n=ms[k] ** 2)
                    ind += ms[k] ** 2
                dims = {'l': ml, 's': ms, 'q': []}
                dinfres = misc.snrm2(rz, dims) / resz0 / -pcost
        return {'status': status, 'x': x, 'sl': sl, 'ss': ss, 'y': y, 'zl': zl, 'zs': zs, 'primal objective': pcost, 'dual objective': dcost, 'gap': gap, 'relative gap': relgap, 'primal infeasibility': pres, 'dual infeasibility': dres, 'residual as primal infeasibility certificate': pinfres, 'residual as dual infeasibility certificate': dinfres, 'primal slack': pslack, 'dual slack': dslack}
    h = matrix(0.0, (N, 1))
    if isinstance(Gl, matrix) or [Gk for Gk in Gs if isinstance(Gk, matrix)]:
        G = matrix(0.0, (N, n))
    else:
        G = spmatrix([], [], [], (N, n), 'd')
    h[:ml] = hl
    G[:ml, :] = Gl
    ind = ml
    for k in range(len(ms)):
        m = ms[k]
        h[ind:ind + m * m] = hs[k][:]
        G[ind:ind + m * m, :] = Gs[k]
        ind += m ** 2
    if primalstart:
        ps = {}
        ps['x'] = primalstart['x']
        ps['s'] = matrix(0.0, (N, 1))
        if ml:
            ps['s'][:ml] = primalstart['sl']
        if ms:
            ind = ml
            for k in range(len(ms)):
                m = ms[k]
                ps['s'][ind:ind + m * m] = primalstart['ss'][k][:]
                ind += m ** 2
    else:
        ps = None
    if dualstart:
        ds = {}
        if p:
            ds['y'] = dualstart['y']
        ds['z'] = matrix(0.0, (N, 1))
        if ml:
            ds['z'][:ml] = dualstart['zl']
        if ms:
            ind = ml
            for k in range(len(ms)):
                m = ms[k]
                ds['z'][ind:ind + m * m] = dualstart['zs'][k][:]
                ind += m ** 2
    else:
        ds = None
    sol = conelp(c, G, h, dims, A=A, b=b, primalstart=ps, dualstart=ds, kktsolver=kktsolver, options=options)
    if sol['s'] is None:
        sol['sl'] = None
        sol['ss'] = None
    else:
        sol['sl'] = sol['s'][:ml]
        sol['ss'] = [matrix(0.0, (mk, mk)) for mk in ms]
        ind = ml
        for k in range(len(ms)):
            m = ms[k]
            sol['ss'][k][:] = sol['s'][ind:ind + m * m]
            ind += m ** 2
    del sol['s']
    if sol['z'] is None:
        sol['zl'] = None
        sol['zs'] = None
    else:
        sol['zl'] = sol['z'][:ml]
        sol['zs'] = [matrix(0.0, (mk, mk)) for mk in ms]
        ind = ml
        for k in range(len(ms)):
            m = ms[k]
            sol['zs'][k][:] = sol['z'][ind:ind + m * m]
            ind += m ** 2
    del sol['z']
    return sol
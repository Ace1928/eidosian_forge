import sys
def conelp(c, G, h, dims=None, A=None, b=None, primalstart=None, dualstart=None, kktsolver=None, xnewcopy=None, xdot=None, xaxpy=None, xscal=None, ynewcopy=None, ydot=None, yaxpy=None, yscal=None, **kwargs):
    """
    Solves a pair of primal and dual cone programs

        minimize    c'*x
        subject to  G*x + s = h
                    A*x = b
                    s >= 0

        maximize    -h'*z - b'*y
        subject to  G'*z + A'*y + c = 0
                    z >= 0.

    The inequalities are with respect to a cone C defined as the Cartesian
    product of N + M + 1 cones:

        C = C_0 x C_1 x .... x C_N x C_{N+1} x ... x C_{N+M}.

    The first cone C_0 is the nonnegative orthant of dimension ml.
    The next N cones are second order cones of dimension mq[0], ...,
    mq[N-1].  The second order cone of dimension m is defined as

        { (u0, u1) in R x R^{m-1} | u0 >= ||u1||_2 }.

    The next M cones are positive semidefinite cones of order ms[0], ...,
    ms[M-1] >= 0.


    Input arguments (basic usage).

        c is a dense 'd' matrix of size (n,1).

        dims is a dictionary with the dimensions of the components of C.
        It has three fields.
        - dims['l'] = ml, the dimension of the nonnegative orthant C_0.
          (ml >= 0.)
        - dims['q'] = mq = [ mq[0], mq[1], ..., mq[N-1] ], a list of N
          integers with the dimensions of the second order cones C_1, ...,
          C_N.  (N >= 0 and mq[k] >= 1.)
        - dims['s'] = ms = [ ms[0], ms[1], ..., ms[M-1] ], a list of M
          integers with the orders of the semidefinite cones C_{N+1}, ...,
          C_{N+M}.  (M >= 0 and ms[k] >= 0.)
        The default value of dims is {'l': G.size[0], 'q': [], 's': []}.

        G is a dense or sparse 'd' matrix of size (K,n), where

            K = ml + mq[0] + ... + mq[N-1] + ms[0]**2 + ... + ms[M-1]**2.

        Each column of G describes a vector

            v = ( v_0, v_1, ..., v_N, vec(v_{N+1}), ..., vec(v_{N+M}) )

        in V = R^ml x R^mq[0] x ... x R^mq[N-1] x S^ms[0] x ... x S^ms[M-1]
        stored as a column vector

            [ v_0; v_1; ...; v_N; vec(v_{N+1}); ...; vec(v_{N+M}) ].

        Here, if u is a symmetric matrix of order m, then vec(u) is the
        matrix u stored in column major order as a vector of length m**2.
        We use BLAS unpacked 'L' storage, i.e., the entries in vec(u)
        corresponding to the strictly upper triangular entries of u are
        not referenced.

        h is a dense 'd' matrix of size (K,1), representing a vector in V,
        in the same format as the columns of G.

        A is a dense or sparse 'd' matrix of size (p,n).  The default value
        is a sparse 'd' matrix of size (0,n).

        b is a dense 'd' matrix of size (p,1).   The default value is a
        dense 'd' matrix of size (0,1).

        The argument primalstart is a dictionary with keys 'x', 's'.  It
        specifies an optional primal starting point.
        - primalstart['x'] is a dense 'd' matrix of size (n,1).
        - primalstart['s'] is a dense 'd' matrix of size (K,1),
          representing a vector that is strictly positive with respect
          to the cone C.

        The argument dualstart is a dictionary with keys 'y', 'z'.  It
        specifies an optional dual starting point.
        - dualstart['y'] is a dense 'd' matrix of size (p,1).
        - dualstart['z'] is a dense 'd' matrix of size (K,1), representing
          a vector that is strictly positive with respect to the cone C.

        It is assumed that rank(A) = p and rank([A; G]) = n.

        The other arguments are normally not needed.  They make it possible
        to exploit certain types of structure, as described below.

    Output arguments.

        Returns a dictionary with keys 'status', 'x', 's', 'z', 'y',
        'primal objective', 'dual objective', 'gap', 'relative gap',
        'primal infeasibility', 'dual infeasibility', 'primal slack',
        'dual slack', 'residual as primal infeasibility certificate',
        'residual as dual infeasibility certificate', 'iterations'.

        The 'status' field has values 'optimal', 'primal infeasible',
        'dual infeasible', or 'unknown'.  The 'iterations' field is the
        number of iterations taken.  The values of the other fields depend
        on the exit status.

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

        - 'primal slack': the smallest primal slack, sup {t | s >= t*e },
           where

              e = ( e_0, e_1, ..., e_N, e_{N+1}, ..., e_{M+N} )

          is the identity vector in C.  e_0 is an ml-vector of ones,
          e_k, k = 1,..., N, are unit vectors (1,0,...,0) of length mq[k],
          and e_k = vec(I) where I is the identity matrix of order ms[k].
        - 'dual slack': the smallest dual slack, sup {t | z >= t*e }.
        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate': None.
        The primal infeasibility is guaranteed to be less than
        solvers.options['feastol'] (default 1e-7).  The dual infeasibility
        is guaranteed to be less than solvers.options['feastol']
        (default 1e-7).  The gap is less than solvers.options['abstol']
        (default 1e-7) or the relative gap is less than
        solvers.options['reltol'] (default 1e-6).

        Status 'primal infeasible'.
        - 'x', 's': None.
        - 'y', 'z' are an approximate certificate of infeasibility

              -h'*z - b'*y = 1,  G'*z + A'*y = 0,  z >= 0.

        - 'primal objective': None.
        - 'dual objective': 1.0.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': None.
        - 'dual slack': the smallest dual slack, sup {t | z >= t*e }.
        - 'residual as primal infeasibility certificate': the residual in
          the condition of the infeasibility certificate, defined as

              || G'*z + A'*y || / max(1, ||c||).

        - 'residual as dual infeasibility certificate': None.
        The residual as primal infeasiblity certificate is guaranteed
        to be less than solvers.options['feastol'] (default 1e-7).

        Status 'dual infeasible'.
        - 'x', 's' are an approximate proof of dual infeasibility

              c'*x = -1,  G*x + s = 0,  A*x = 0,  s >= 0.

        - 'y', 'z': None.
        - 'primal objective': -1.0.
        - 'dual objective': None.
        - 'gap', 'relative gap': None.
        - 'primal infeasibility' and 'dual infeasibility': None.
        - 'primal slack': the smallest primal slack, sup {t | s >= t*e}.
        - 'dual slack': None.
        - 'residual as primal infeasibility certificate': None.
        - 'residual as dual infeasibility certificate: the residual in
          the conditions of the infeasibility certificate, defined as
          the maximum of

              || G*x + s || / max(1, ||h||) and || A*x || / max(1, ||b||).

        The residual as dual infeasiblity certificate is guaranteed
        to be less than solvers.options['feastol'] (default 1e-7).

        Status 'unknown'.
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

        - 'primal slack': the smallest primal slack, sup {t | s >= t*e}.
        - 'dual slack': the smallest dual slack, sup {t | z >= t*e}.
        - 'residual as primal infeasibility certificate': None if
           h'*z + b'*y >= 0; the residual

              || G'*z + A'*y || / ( -(h'*z + b'*y) * max(1, ||c||) )

          otherwise.
        - 'residual as dual infeasibility certificate':
          None if c'*x >= 0; the maximum of the residuals

              || G*x + s || / ( -c'*x * max(1, ||h||) )

          and

              || A*x || / ( -c'*x * max(1, ||b||) )

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


    Advanced usage.

        Three mechanisms are provided to express problem structure.

        1.  The user can provide a customized routine for solving linear
        equations (`KKT systems')

            [ 0  A'  G'   ] [ ux ]   [ bx ]
            [ A  0   0    ] [ uy ] = [ by ].
            [ G  0  -W'*W ] [ uz ]   [ bz ]

        W is a scaling matrix, a block diagonal mapping

           W*z = ( W0*z_0, ..., W_{N+M}*z_{N+M} )

        defined as follows.

        - For the 'l' block (W_0):

              W_0 = diag(d),

          with d a positive vector of length ml.

        - For the 'q' blocks (W_{k+1}, k = 0, ..., N-1):

              W_{k+1} = beta_k * ( 2 * v_k * v_k' - J )

          where beta_k is a positive scalar, v_k is a vector in R^mq[k]
          with v_k[0] > 0 and v_k'*J*v_k = 1, and J = [1, 0; 0, -I].

        - For the 's' blocks (W_{k+N}, k = 0, ..., M-1):

              W_k * x = vec(r_k' * mat(x) * r_k)

          where r_k is a nonsingular matrix of order ms[k], and mat(x) is
          the inverse of the vec operation.

        The optional argument kktsolver is a Python function that will be
        called as f = kktsolver(W), where W is a dictionary that contains
        the parameters of the scaling:

        - W['d'] is a positive 'd' matrix of size (ml,1).
        - W['di'] is a positive 'd' matrix with the elementwise inverse of
          W['d'].
        - W['beta'] is a list [ beta_0, ..., beta_{N-1} ]
        - W['v'] is a list [ v_0, ..., v_{N-1} ]
        - W['r'] is a list [ r_0, ..., r_{M-1} ]
        - W['rti'] is a list [ rti_0, ..., rti_{M-1} ], with rti_k the
          inverse of the transpose of r_k.

        The call f = kktsolver(W) should return a function f that solves
        the KKT system by f(x, y, z).  On entry, x, y, z contain the
        righthand side bx, by, bz.  On exit, they contain the solution,
        with uz scaled: the argument z contains W*uz.  In other words,
        on exit, x, y, z are the solution of

            [ 0  A'  G'*W^{-1} ] [ ux ]   [ bx ]
            [ A  0   0         ] [ uy ] = [ by ].
            [ G  0  -W'        ] [ uz ]   [ bz ]


        2.  The linear operators G*u and A*u can be specified by providing
        Python functions instead of matrices.  This can only be done in
        combination with 1. above, i.e., it requires the kktsolver
        argument.

        If G is a function, the call G(u, v, alpha, beta, trans)
        should evaluate the matrix-vector products

            v := alpha * G * u + beta * v  if trans is 'N'
            v := alpha * G' * u + beta * v  if trans is 'T'.

        The arguments u and v are required.  The other arguments have
        default values alpha = 1.0, beta = 0.0, trans = 'N'.

        If A is a function, the call A(u, v, alpha, beta, trans) should
        evaluate the matrix-vectors products

            v := alpha * A * u + beta * v if trans is 'N'
            v := alpha * A' * u + beta * v if trans is 'T'.

        The arguments u and v are required.  The other arguments
        have default values alpha = 1.0, beta = 0.0, trans = 'N'.


        3.  Instead of using the default representation of the primal
        variable x and the dual variable y as one-column 'd' matrices,
        we can represent these variables and the corresponding parameters
        c and b by arbitrary Python objects (matrices, lists, dictionaries,
        etc.).  This can only be done in combination with 1. and 2. above,
        i.e., it requires a user-provided KKT solver and an operator
        description of the linear mappings.  It also requires the arguments
        xnewcopy, xdot, xscal, xaxpy, ynewcopy, ydot, yscal, yaxpy.  These
        arguments are functions defined as follows.

        If X is the vector space of primal variables x, then:
        - xnewcopy(u) creates a new copy of the vector u in X.
        - xdot(u, v) returns the inner product of two vectors u and v in X.
        - xscal(alpha, u) computes u := alpha*u, where alpha is a scalar
          and u is a vector in X.
        - xaxpy(u, v, alpha = 1.0) computes v := alpha*u + v for a scalar
          alpha and two vectors u and v in X.
        If this option is used, the argument c must be in the same format
        as x, the argument G must be a Python function, the argument A
        must be a Python function or None, and the argument kktsolver is
        required.

        If Y is the vector space of primal variables y:
        - ynewcopy(u) creates a new copy of the vector u in Y.
        - ydot(u, v) returns the inner product of two vectors u and v in Y.
        - yscal(alpha, u) computes u := alpha*u, where alpha is a scalar
          and u is a vector in Y.
        - yaxpy(u, v, alpha = 1.0) computes v := alpha*u + v for a scalar
          alpha and two vectors u and v in Y.
        If this option is used, the argument b must be in the same format
        as y, the argument A must be a Python function or None, and the
        argument kktsolver is required.


    Control parameters.

        The following control parameters can be modified by adding an
        entry to the dictionary options.

        options['show_progress'] True/False (default: True)
        options['maxiters'] positive integer (default: 100)
        options['refinement'] positive integer (default: 0 for problems
            with no second-order cone and matrix inequality constraints;
            1 otherwise)
        options['abstol'] scalar (default: 1e-7 )
        options['reltol'] scalar (default: 1e-6)
        options['feastol'] scalar (default: 1e-7).

    """
    import math
    from cvxopt import base, blas, misc, matrix, spmatrix
    EXPON = 3
    STEP = 0.99
    options = kwargs.get('options', globals()['options'])
    DEBUG = options.get('debug', False)
    KKTREG = options.get('kktreg', None)
    if KKTREG is None:
        pass
    elif not isinstance(KKTREG, (float, int, long)) or KKTREG < 0.0:
        raise ValueError("options['kktreg'] must be a nonnegative scalar")
    MAXITERS = options.get('maxiters', 100)
    if not isinstance(MAXITERS, (int, long)) or MAXITERS < 1:
        raise ValueError("options['maxiters'] must be a positive integer")
    ABSTOL = options.get('abstol', 1e-07)
    if not isinstance(ABSTOL, (float, int, long)):
        raise ValueError("options['abstol'] must be a scalar")
    RELTOL = options.get('reltol', 1e-06)
    if not isinstance(RELTOL, (float, int, long)):
        raise ValueError("options['reltol'] must be a scalar")
    if RELTOL <= 0.0 and ABSTOL <= 0.0:
        raise ValueError("at least one of options['reltol'] and options['abstol'] must be positive")
    FEASTOL = options.get('feastol', 1e-07)
    if not isinstance(FEASTOL, (float, int, long)) or FEASTOL <= 0.0:
        raise ValueError("options['feastol'] must be a positive scalar")
    show_progress = options.get('show_progress', True)
    if kktsolver is None:
        if dims and (dims['q'] or dims['s']):
            kktsolver = 'qr'
        else:
            kktsolver = 'chol2'
    defaultsolvers = ('ldl', 'ldl2', 'qr', 'chol', 'chol2')
    if isinstance(kktsolver, str) and kktsolver not in defaultsolvers:
        raise ValueError("'%s' is not a valid value for kktsolver" % kktsolver)
    customkkt = not isinstance(kktsolver, str)
    matrixG = isinstance(G, (matrix, spmatrix))
    matrixA = isinstance(A, (matrix, spmatrix))
    if (not matrixG or (not matrixA and A is not None)) and (not customkkt):
        raise ValueError('use of function valued G, A requires a user-provided kktsolver')
    customx = xnewcopy != None or xdot != None or xaxpy != None or (xscal != None)
    if customx and (matrixG or matrixA or (not customkkt)):
        raise ValueError('use of non-vector type for x requires function valued G, A and user-provided kktsolver')
    customy = ynewcopy != None or ydot != None or yaxpy != None or (yscal != None)
    if customy and (matrixA or not customkkt):
        raise ValueError('use of non-vector type for y requires function valued A and user-provided kktsolver')
    if not customx and (not isinstance(c, matrix) or c.typecode != 'd' or c.size[1] != 1):
        raise TypeError("'c' must be a 'd' matrix with one column")
    if not isinstance(h, matrix) or h.typecode != 'd' or h.size[1] != 1:
        raise TypeError("'h' must be a 'd' matrix with 1 column")
    if not dims:
        dims = {'l': h.size[0], 'q': [], 's': []}
    if not isinstance(dims['l'], (int, long)) or dims['l'] < 0:
        raise TypeError("'dims['l']' must be a nonnegative integer")
    if [k for k in dims['q'] if not isinstance(k, (int, long)) or k < 1]:
        raise TypeError("'dims['q']' must be a list of positive integers")
    if [k for k in dims['s'] if not isinstance(k, (int, long)) or k < 0]:
        raise TypeError("'dims['s']' must be a list of nonnegative integers")
    refinement = options.get('refinement', None)
    if refinement is None:
        if dims['q'] or dims['s']:
            refinement = 1
        else:
            refinement = 0
    elif not isinstance(refinement, (int, long)) or refinement < 0:
        raise ValueError("options['refinement'] must be a nonnegative integer")
    cdim = dims['l'] + sum(dims['q']) + sum([k ** 2 for k in dims['s']])
    cdim_pckd = dims['l'] + sum(dims['q']) + sum([k * (k + 1) / 2 for k in dims['s']])
    cdim_diag = dims['l'] + sum(dims['q']) + sum(dims['s'])
    if h.size[0] != cdim:
        raise TypeError("'h' must be a 'd' matrix of size (%d,1)" % cdim)
    indq = [dims['l']]
    for k in dims['q']:
        indq = indq + [indq[-1] + k]
    inds = [indq[-1]]
    for k in dims['s']:
        inds = inds + [inds[-1] + k ** 2]
    if matrixG:
        if G.typecode != 'd' or G.size != (cdim, c.size[0]):
            raise TypeError("'G' must be a 'd' matrix of size (%d, %d)" % (cdim, c.size[0]))

        def Gf(x, y, trans='N', alpha=1.0, beta=0.0):
            misc.sgemv(G, x, y, dims, trans=trans, alpha=alpha, beta=beta)
    else:
        Gf = G
    if A is None:
        if customx or customy:

            def A(x, y, trans='N', alpha=1.0, beta=0.0):
                if trans == 'N':
                    pass
                else:
                    xscal(beta, y)
        else:
            A = spmatrix([], [], [], (0, c.size[0]))
            matrixA = True
    if matrixA:
        if A.typecode != 'd' or A.size[1] != c.size[0]:
            raise TypeError("'A' must be a 'd' matrix with %d columns " % c.size[0])

        def Af(x, y, trans='N', alpha=1.0, beta=0.0):
            base.gemv(A, x, y, trans=trans, alpha=alpha, beta=beta)
    else:
        Af = A
    if not customy:
        if b is None:
            b = matrix(0.0, (0, 1))
        if not isinstance(b, matrix) or b.typecode != 'd' or b.size[1] != 1:
            raise TypeError("'b' must be a 'd' matrix with one column")
        if matrixA and b.size[0] != A.size[0]:
            raise TypeError("'b' must have length %d" % A.size[0])
    elif b is None:
        raise ValueError('use of non vector type for y requires b')
    if kktsolver in defaultsolvers:
        if KKTREG is None and (b.size[0] > c.size[0] or b.size[0] + cdim_pckd < c.size[0]):
            raise ValueError('Rank(A) < p or Rank([G; A]) < n')
        if kktsolver == 'ldl':
            factor = misc.kkt_ldl(G, dims, A, kktreg=KKTREG)
        elif kktsolver == 'ldl2':
            factor = misc.kkt_ldl2(G, dims, A)
        elif kktsolver == 'qr':
            factor = misc.kkt_qr(G, dims, A)
        elif kktsolver == 'chol':
            factor = misc.kkt_chol(G, dims, A)
        else:
            factor = misc.kkt_chol2(G, dims, A)

        def kktsolver(W):
            return factor(W)
    ws3, wz3 = (matrix(0.0, (cdim, 1)), matrix(0.0, (cdim, 1)))

    def res(ux, uy, uz, utau, us, ukappa, vx, vy, vz, vtau, vs, vkappa, W, dg, lmbda):
        Af(uy, vx, alpha=-1.0, beta=1.0, trans='T')
        blas.copy(uz, wz3)
        misc.scale(wz3, W, inverse='I')
        Gf(wz3, vx, alpha=-1.0, beta=1.0, trans='T')
        xaxpy(c, vx, alpha=-utau[0] / dg)
        Af(ux, vy, alpha=1.0, beta=1.0)
        yaxpy(b, vy, alpha=-utau[0] / dg)
        Gf(ux, vz, alpha=1.0, beta=1.0)
        blas.axpy(h, vz, alpha=-utau[0] / dg)
        blas.copy(us, ws3)
        misc.scale(ws3, W, trans='T')
        blas.axpy(ws3, vz)
        vtau[0] += dg * ukappa[0] + xdot(c, ux) + ydot(b, uy) + misc.sdot(h, wz3, dims)
        blas.copy(us, ws3)
        blas.axpy(uz, ws3)
        misc.sprod(ws3, lmbda, dims, diag='D')
        blas.axpy(ws3, vs)
        vkappa[0] += lmbda[-1] * (utau[0] + ukappa[0])
    if xnewcopy is None:
        xnewcopy = matrix
    if xdot is None:
        xdot = blas.dot
    if xaxpy is None:
        xaxpy = blas.axpy
    if xscal is None:
        xscal = blas.scal

    def xcopy(x, y):
        xscal(0.0, y)
        xaxpy(x, y)
    if ynewcopy is None:
        ynewcopy = matrix
    if ydot is None:
        ydot = blas.dot
    if yaxpy is None:
        yaxpy = blas.axpy
    if yscal is None:
        yscal = blas.scal

    def ycopy(x, y):
        yscal(0.0, y)
        yaxpy(x, y)
    resx0 = max(1.0, math.sqrt(xdot(c, c)))
    resy0 = max(1.0, math.sqrt(ydot(b, b)))
    resz0 = max(1.0, misc.snrm2(h, dims))
    x = xnewcopy(c)
    xscal(0.0, x)
    y = ynewcopy(b)
    yscal(0.0, y)
    s, z = (matrix(0.0, (cdim, 1)), matrix(0.0, (cdim, 1)))
    dx, dy = (xnewcopy(c), ynewcopy(b))
    ds, dz = (matrix(0.0, (cdim, 1)), matrix(0.0, (cdim, 1)))
    dkappa, dtau = (matrix(0.0, (1, 1)), matrix(0.0, (1, 1)))
    if primalstart is None or dualstart is None:
        W = {}
        W['d'] = matrix(1.0, (dims['l'], 1))
        W['di'] = matrix(1.0, (dims['l'], 1))
        W['v'] = [matrix(0.0, (m, 1)) for m in dims['q']]
        W['beta'] = len(dims['q']) * [1.0]
        for v in W['v']:
            v[0] = 1.0
        W['r'] = [matrix(0.0, (m, m)) for m in dims['s']]
        W['rti'] = [matrix(0.0, (m, m)) for m in dims['s']]
        for r in W['r']:
            r[::r.size[0] + 1] = 1.0
        for rti in W['rti']:
            rti[::rti.size[0] + 1] = 1.0
        try:
            f = kktsolver(W)
        except ArithmeticError:
            raise ValueError('Rank(A) < p or Rank([G; A]) < n')
    if primalstart is None:
        xscal(0.0, x)
        ycopy(b, dy)
        blas.copy(h, s)
        try:
            f(x, dy, s)
        except ArithmeticError:
            raise ValueError('Rank(A) < p or Rank([G; A]) < n')
        blas.scal(-1.0, s)
    else:
        xcopy(primalstart['x'], x)
        blas.copy(primalstart['s'], s)
    ts = misc.max_step(s, dims)
    if ts >= 0 and primalstart:
        raise ValueError('initial s is not positive')
    if dualstart is None:
        xcopy(c, dx)
        xscal(-1.0, dx)
        yscal(0.0, y)
        blas.scal(0.0, z)
        try:
            f(dx, y, z)
        except ArithmeticError:
            raise ValueError('Rank(A) < p or Rank([G; A]) < n')
    else:
        if 'y' in dualstart:
            ycopy(dualstart['y'], y)
        blas.copy(dualstart['z'], z)
    tz = misc.max_step(z, dims)
    if tz >= 0 and dualstart:
        raise ValueError('initial z is not positive')
    nrms = misc.snrm2(s, dims)
    nrmz = misc.snrm2(z, dims)
    if primalstart is None and dualstart is None:
        gap = misc.sdot(s, z, dims)
        pcost = xdot(c, x)
        dcost = -ydot(b, y) - misc.sdot(h, z, dims)
        if pcost < 0.0:
            relgap = gap / -pcost
        elif dcost > 0.0:
            relgap = gap / dcost
        else:
            relgap = None
        if (ts <= 0 and tz <= 0 and (gap <= ABSTOL or (relgap is not None and relgap <= RELTOL))) and KKTREG is None:
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                misc.symm(s, m, ind)
                misc.symm(z, m, ind)
                ind += m ** 2
            rx = xnewcopy(c)
            Af(y, rx, beta=1.0, trans='T')
            Gf(z, rx, beta=1.0, trans='T')
            resx = math.sqrt(xdot(rx, rx))
            ry = ynewcopy(b)
            Af(x, ry, alpha=-1.0, beta=1.0)
            resy = math.sqrt(ydot(ry, ry))
            rz = matrix(0.0, (cdim, 1))
            Gf(x, rz)
            blas.axpy(s, rz)
            blas.axpy(h, rz, alpha=-1.0)
            resz = misc.snrm2(rz, dims)
            pres = max(resy / resy0, resz / resz0)
            dres = resx / resx0
            cx, by, hz = (xdot(c, x), ydot(b, y), misc.sdot(h, z, dims))
            if show_progress:
                print('Optimal solution found.')
            return {'x': x, 'y': y, 's': s, 'z': z, 'status': 'optimal', 'gap': gap, 'relative gap': relgap, 'primal objective': cx, 'dual objective': -(by + hz), 'primal infeasibility': pres, 'primal slack': -ts, 'dual slack': -tz, 'dual infeasibility': dres, 'residual as primal infeasibility certificate': None, 'residual as dual infeasibility certificate': None, 'iterations': 0}
        if ts >= -1e-08 * max(nrms, 1.0):
            a = 1.0 + ts
            s[:dims['l']] += a
            s[indq[:-1]] += a
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                s[ind:ind + m * m:m + 1] += a
                ind += m ** 2
        if tz >= -1e-08 * max(nrmz, 1.0):
            a = 1.0 + tz
            z[:dims['l']] += a
            z[indq[:-1]] += a
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                z[ind:ind + m * m:m + 1] += a
                ind += m ** 2
    elif primalstart is None and dualstart is not None:
        if ts >= -1e-08 * max(nrms, 1.0):
            a = 1.0 + ts
            s[:dims['l']] += a
            s[indq[:-1]] += a
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                s[ind:ind + m * m:m + 1] += a
                ind += m ** 2
    elif primalstart is not None and dualstart is None:
        if tz >= -1e-08 * max(nrmz, 1.0):
            a = 1.0 + tz
            z[:dims['l']] += a
            z[indq[:-1]] += a
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                z[ind:ind + m * m:m + 1] += a
                ind += m ** 2
    tau, kappa = (1.0, 1.0)
    rx, hrx = (xnewcopy(c), xnewcopy(c))
    ry, hry = (ynewcopy(b), ynewcopy(b))
    rz, hrz = (matrix(0.0, (cdim, 1)), matrix(0.0, (cdim, 1)))
    sigs = matrix(0.0, (sum(dims['s']), 1))
    sigz = matrix(0.0, (sum(dims['s']), 1))
    lmbda = matrix(0.0, (cdim_diag + 1, 1))
    lmbdasq = matrix(0.0, (cdim_diag + 1, 1))
    gap = misc.sdot(s, z, dims)
    for iters in range(MAXITERS + 1):
        Af(y, hrx, alpha=-1.0, trans='T')
        Gf(z, hrx, alpha=-1.0, beta=1.0, trans='T')
        hresx = math.sqrt(xdot(hrx, hrx))
        xcopy(hrx, rx)
        xaxpy(c, rx, alpha=-tau)
        resx = math.sqrt(xdot(rx, rx)) / tau
        Af(x, hry)
        hresy = math.sqrt(ydot(hry, hry))
        ycopy(hry, ry)
        yaxpy(b, ry, alpha=-tau)
        resy = math.sqrt(ydot(ry, ry)) / tau
        Gf(x, hrz)
        blas.axpy(s, hrz)
        hresz = misc.snrm2(hrz, dims)
        blas.scal(0, rz)
        blas.axpy(hrz, rz)
        blas.axpy(h, rz, alpha=-tau)
        resz = misc.snrm2(rz, dims) / tau
        cx, by, hz = (xdot(c, x), ydot(b, y), misc.sdot(h, z, dims))
        rt = kappa + cx + by + hz
        pcost, dcost = (cx / tau, -(by + hz) / tau)
        if pcost < 0.0:
            relgap = gap / -pcost
        elif dcost > 0.0:
            relgap = gap / dcost
        else:
            relgap = None
        pres = max(resy / resy0, resz / resz0)
        dres = resx / resx0
        if hz + by < 0.0:
            pinfres = hresx / resx0 / (-hz - by)
        else:
            pinfres = None
        if cx < 0.0:
            dinfres = max(hresy / resy0, hresz / resz0) / -cx
        else:
            dinfres = None
        if show_progress:
            if iters == 0:
                print('% 10s% 12s% 10s% 8s% 7s % 5s' % ('pcost', 'dcost', 'gap', 'pres', 'dres', 'k/t'))
            print('%2d: % 8.4e % 8.4e % 4.0e% 7.0e% 7.0e% 7.0e' % (iters, pcost, dcost, gap, pres, dres, kappa / tau))
        if pres <= FEASTOL and dres <= FEASTOL and (gap <= ABSTOL or (relgap is not None and relgap <= RELTOL)) or iters == MAXITERS:
            xscal(1.0 / tau, x)
            yscal(1.0 / tau, y)
            blas.scal(1.0 / tau, s)
            blas.scal(1.0 / tau, z)
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                misc.symm(s, m, ind)
                misc.symm(z, m, ind)
                ind += m ** 2
            ts = misc.max_step(s, dims)
            tz = misc.max_step(z, dims)
            if iters == MAXITERS:
                if show_progress:
                    print('Terminated (maximum number of iterations reached).')
                return {'x': x, 'y': y, 's': s, 'z': z, 'status': 'unknown', 'gap': gap, 'relative gap': relgap, 'primal objective': pcost, 'dual objective': dcost, 'primal infeasibility': pres, 'dual infeasibility': dres, 'primal slack': -ts, 'dual slack': -tz, 'residual as primal infeasibility certificate': pinfres, 'residual as dual infeasibility certificate': dinfres, 'iterations': iters}
            else:
                if show_progress:
                    print('Optimal solution found.')
                return {'x': x, 'y': y, 's': s, 'z': z, 'status': 'optimal', 'gap': gap, 'relative gap': relgap, 'primal objective': pcost, 'dual objective': dcost, 'primal infeasibility': pres, 'dual infeasibility': dres, 'primal slack': -ts, 'dual slack': -tz, 'residual as primal infeasibility certificate': None, 'residual as dual infeasibility certificate': None, 'iterations': iters}
        elif pinfres is not None and pinfres <= FEASTOL:
            yscal(1.0 / (-hz - by), y)
            blas.scal(1.0 / (-hz - by), z)
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                misc.symm(z, m, ind)
                ind += m ** 2
            tz = misc.max_step(z, dims)
            if show_progress:
                print('Certificate of primal infeasibility found.')
            return {'x': None, 'y': y, 's': None, 'z': z, 'status': 'primal infeasible', 'gap': None, 'relative gap': None, 'primal objective': None, 'dual objective': 1.0, 'primal infeasibility': None, 'dual infeasibility': None, 'primal slack': None, 'dual slack': -tz, 'residual as primal infeasibility certificate': pinfres, 'residual as dual infeasibility certificate': None, 'iterations': iters}
        elif dinfres is not None and dinfres <= FEASTOL:
            xscal(1.0 / -cx, x)
            blas.scal(1.0 / -cx, s)
            ind = dims['l'] + sum(dims['q'])
            for m in dims['s']:
                misc.symm(s, m, ind)
                ind += m ** 2
            y, z = (None, None)
            ts = misc.max_step(s, dims)
            if show_progress:
                print('Certificate of dual infeasibility found.')
            return {'x': x, 'y': None, 's': s, 'z': None, 'status': 'dual infeasible', 'gap': None, 'relative gap': None, 'primal objective': -1.0, 'dual objective': None, 'primal infeasibility': None, 'dual infeasibility': None, 'primal slack': -ts, 'dual slack': None, 'residual as primal infeasibility certificate': None, 'residual as dual infeasibility certificate': dinfres, 'iterations': iters}
        if iters == 0:
            W = misc.compute_scaling(s, z, lmbda, dims, mnl=0)
            dg = math.sqrt(kappa / tau)
            dgi = math.sqrt(tau / kappa)
            lmbda[-1] = math.sqrt(tau * kappa)
        misc.ssqr(lmbdasq, lmbda, dims)
        lmbdasq[-1] = lmbda[-1] ** 2
        try:
            f3 = kktsolver(W)
            if iters == 0:
                x1, y1 = (xnewcopy(c), ynewcopy(b))
                z1 = matrix(0.0, (cdim, 1))
            xcopy(c, x1)
            xscal(-1, x1)
            ycopy(b, y1)
            blas.copy(h, z1)
            f3(x1, y1, z1)
            xscal(dgi, x1)
            yscal(dgi, y1)
            blas.scal(dgi, z1)
        except ArithmeticError:
            if iters == 0 and primalstart and dualstart:
                raise ValueError('Rank(A) < p or Rank([G; A]) < n')
            else:
                xscal(1.0 / tau, x)
                yscal(1.0 / tau, y)
                blas.scal(1.0 / tau, s)
                blas.scal(1.0 / tau, z)
                ind = dims['l'] + sum(dims['q'])
                for m in dims['s']:
                    misc.symm(s, m, ind)
                    misc.symm(z, m, ind)
                    ind += m ** 2
                ts = misc.max_step(s, dims)
                tz = misc.max_step(z, dims)
                if show_progress:
                    print('Terminated (singular KKT matrix).')
                return {'x': x, 'y': y, 's': s, 'z': z, 'status': 'unknown', 'gap': gap, 'relative gap': relgap, 'primal objective': pcost, 'dual objective': dcost, 'primal infeasibility': pres, 'dual infeasibility': dres, 'primal slack': -ts, 'dual slack': -tz, 'residual as primal infeasibility certificate': pinfres, 'residual as dual infeasibility certificate': dinfres, 'iterations': iters}
        if iters == 0:
            th = matrix(0.0, (cdim, 1))
        blas.copy(h, th)
        misc.scale(th, W, trans='T', inverse='I')

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
        if iters == 0:
            if refinement or DEBUG:
                wx, wy = (xnewcopy(c), ynewcopy(b))
                wz, ws = (matrix(0.0, (cdim, 1)), matrix(0.0, (cdim, 1)))
                wtau, wkappa = (matrix(0.0), matrix(0.0))
            if refinement:
                wx2, wy2 = (xnewcopy(c), ynewcopy(b))
                wz2, ws2 = (matrix(0.0, (cdim, 1)), matrix(0.0, (cdim, 1)))
                wtau2, wkappa2 = (matrix(0.0), matrix(0.0))

        def f6(x, y, z, tau, s, kappa):
            if refinement or DEBUG:
                xcopy(x, wx)
                ycopy(y, wy)
                blas.copy(z, wz)
                wtau[0] = tau[0]
                blas.copy(s, ws)
                wkappa[0] = kappa[0]
            f6_no_ir(x, y, z, tau, s, kappa)
            for i in range(refinement):
                xcopy(wx, wx2)
                ycopy(wy, wy2)
                blas.copy(wz, wz2)
                wtau2[0] = wtau[0]
                blas.copy(ws, ws2)
                wkappa2[0] = wkappa[0]
                res(x, y, z, tau, s, kappa, wx2, wy2, wz2, wtau2, ws2, wkappa2, W, dg, lmbda)
                f6_no_ir(wx2, wy2, wz2, wtau2, ws2, wkappa2)
                xaxpy(wx2, x)
                yaxpy(wy2, y)
                blas.axpy(wz2, z)
                tau[0] += wtau2[0]
                blas.axpy(ws2, s)
                kappa[0] += wkappa2[0]
            if DEBUG:
                res(x, y, z, tau, s, kappa, wx, wy, wz, wtau, ws, wkappa, W, dg, lmbda)
                print('KKT residuals')
                print("    'x': %e" % math.sqrt(xdot(wx, wx)))
                print("    'y': %e" % math.sqrt(ydot(wy, wy)))
                print("    'z': %e" % misc.snrm2(wz, dims))
                print("    'tau': %e" % abs(wtau[0]))
                print("    's': %e" % misc.snrm2(ws, dims))
                print("    'kappa': %e" % abs(wkappa[0]))
        mu = blas.nrm2(lmbda) ** 2 / (1 + cdim_diag)
        sigma = 0.0
        for i in [0, 1]:
            blas.copy(lmbdasq, ds, n=dims['l'] + sum(dims['q']))
            ind = dims['l'] + sum(dims['q'])
            ind2 = ind
            blas.scal(0.0, ds, offset=ind)
            for m in dims['s']:
                blas.copy(lmbdasq, ds, n=m, offsetx=ind2, offsety=ind, incy=m + 1)
                ind += m * m
                ind2 += m
            dkappa[0] = lmbdasq[-1]
            if i == 1:
                blas.axpy(ws3, ds)
                ds[:dims['l']] -= sigma * mu
                ds[indq[:-1]] -= sigma * mu
                ind = dims['l'] + sum(dims['q'])
                ind2 = ind
                for m in dims['s']:
                    ds[ind:ind + m * m:m + 1] -= sigma * mu
                    ind += m * m
                dkappa[0] += wkappa3 - sigma * mu
            xcopy(rx, dx)
            xscal(1.0 - sigma, dx)
            ycopy(ry, dy)
            yscal(1.0 - sigma, dy)
            blas.copy(rz, dz)
            blas.scal(1.0 - sigma, dz)
            dtau[0] = (1.0 - sigma) * rt
            f6(dx, dy, dz, dtau, ds, dkappa)
            if i == 0:
                blas.copy(ds, ws3)
                misc.sprod(ws3, dz, dims)
                wkappa3 = dtau[0] * dkappa[0]
            misc.scale2(lmbda, ds, dims)
            misc.scale2(lmbda, dz, dims)
            if i == 0:
                ts = misc.max_step(ds, dims)
                tz = misc.max_step(dz, dims)
            else:
                ts = misc.max_step(ds, dims, sigma=sigs)
                tz = misc.max_step(dz, dims, sigma=sigz)
            tt = -dtau[0] / lmbda[-1]
            tk = -dkappa[0] / lmbda[-1]
            t = max([0.0, ts, tz, tt, tk])
            if t == 0.0:
                step = 1.0
            elif i == 0:
                step = min(1.0, 1.0 / t)
            else:
                step = min(1.0, STEP / t)
            if i == 0:
                sigma = (1.0 - step) ** EXPON
        xaxpy(dx, x, alpha=step)
        yaxpy(dy, y, alpha=step)
        blas.scal(step, ds, n=dims['l'] + sum(dims['q']))
        blas.scal(step, dz, n=dims['l'] + sum(dims['q']))
        ds[:dims['l']] += 1.0
        dz[:dims['l']] += 1.0
        ds[indq[:-1]] += 1.0
        dz[indq[:-1]] += 1.0
        misc.scale2(lmbda, ds, dims, inverse='I')
        misc.scale2(lmbda, dz, dims, inverse='I')
        blas.scal(step, sigs)
        blas.scal(step, sigz)
        sigs += 1.0
        sigz += 1.0
        blas.tbsv(lmbda, sigs, n=sum(dims['s']), k=0, ldA=1, offsetA=dims['l'] + sum(dims['q']))
        blas.tbsv(lmbda, sigz, n=sum(dims['s']), k=0, ldA=1, offsetA=dims['l'] + sum(dims['q']))
        ind2, ind3 = (dims['l'] + sum(dims['q']), 0)
        for k in range(len(dims['s'])):
            m = dims['s'][k]
            for i in range(m):
                blas.scal(math.sqrt(sigs[ind3 + i]), ds, offset=ind2 + m * i, n=m)
                blas.scal(math.sqrt(sigz[ind3 + i]), dz, offset=ind2 + m * i, n=m)
            ind2 += m * m
            ind3 += m
        misc.update_scaling(W, lmbda, ds, dz)
        dg *= math.sqrt(1.0 - step * tk) / math.sqrt(1.0 - step * tt)
        dgi = 1.0 / dg
        lmbda[-1] *= math.sqrt(1.0 - step * tt) * math.sqrt(1.0 - step * tk)
        blas.copy(lmbda, s, n=dims['l'] + sum(dims['q']))
        ind = dims['l'] + sum(dims['q'])
        ind2 = ind
        for m in dims['s']:
            blas.scal(0.0, s, offset=ind2)
            blas.copy(lmbda, s, offsetx=ind, offsety=ind2, n=m, incy=m + 1)
            ind += m
            ind2 += m * m
        misc.scale(s, W, trans='T')
        blas.copy(lmbda, z, n=dims['l'] + sum(dims['q']))
        ind = dims['l'] + sum(dims['q'])
        ind2 = ind
        for m in dims['s']:
            blas.scal(0.0, z, offset=ind2)
            blas.copy(lmbda, z, offsetx=ind, offsety=ind2, n=m, incy=m + 1)
            ind += m
            ind2 += m * m
        misc.scale(z, W, inverse='I')
        kappa, tau = (lmbda[-1] / dgi, lmbda[-1] * dgi)
        gap = (blas.nrm2(lmbda, n=lmbda.size[0] - 1) / tau) ** 2
import mosek
from cvxopt import matrix, spmatrix, sparse
import sys
def ilp(c, G, h, A=None, b=None, I=None, taskfile=None, **kwargs):
    """
    Solves the mixed integer LP

        minimize    c'*x
        subject to  G*x + s = h
                    A*x = b
                    s >= 0
                    xi integer, forall i in I

    using MOSEK 8.

    solsta, x = ilp(c, G, h, A=None, b=None, I=None, taskfile=None).

    Input arguments

        G is m x n, h is m x 1, A is p x n, b is p x 1.  G and A must be
        dense or sparse 'd' matrices.   h and b are dense 'd' matrices
        with one column.  The default values for A and b are empty
        matrices with zero rows.

        I is a Python set with indices of integer elements of x.  By
        default all elements in x are constrained to be integer, i.e.,
        the default value of I is I = set(range(n))

        Dual variables are not returned for MOSEK.

        Optionally, the interface can write a .task file, required for
        support questions on the MOSEK solver.

    Return values

        solsta is a MOSEK solution status key.

            If solsta is mosek.solsta.integer_optimal, then x contains
                the solution.
            If solsta is mosek.solsta.unknown, then x is None.

            Other return values for solsta include:
                mosek.solsta.near_integer_optimal
            in which case the x value may not be well-defined,
            c.f., section 17.48 of the MOSEK Python API manual.

        x is the solution

    Options are passed to MOSEK solvers via the msk.options dictionary,
    e.g., the following turns off output from the MOSEK solvers

    >>> msk.options = {mosek.iparam.log: 0}

    see the MOSEK Python API manual.
    """
    with mosek.Env() as env:
        if type(c) is not matrix or c.typecode != 'd' or c.size[1] != 1:
            raise TypeError("'c' must be a dense column matrix")
        n = c.size[0]
        if n < 1:
            raise ValueError('number of variables must be at least 1')
        if type(G) is not matrix and type(G) is not spmatrix or G.typecode != 'd' or G.size[1] != n:
            raise TypeError("'G' must be a dense or sparse 'd' matrix with %d columns" % n)
        m = G.size[0]
        if m == 0:
            raise ValueError('m cannot be 0')
        if type(h) is not matrix or h.typecode != 'd' or h.size != (m, 1):
            raise TypeError("'h' must be a 'd' matrix of size (%d,1)" % m)
        if A is None:
            A = spmatrix([], [], [], (0, n), 'd')
        if type(A) is not matrix and type(A) is not spmatrix or A.typecode != 'd' or A.size[1] != n:
            raise TypeError("'A' must be a dense or sparse 'd' matrix with %d columns" % n)
        p = A.size[0]
        if b is None:
            b = matrix(0.0, (0, 1))
        if type(b) is not matrix or b.typecode != 'd' or b.size != (p, 1):
            raise TypeError("'b' must be a dense matrix of size (%d,1)" % p)
        if I is None:
            I = set(range(n))
        if type(I) is not set:
            raise TypeError('invalid argument for integer index set')
        for i in I:
            if type(i) is not int:
                raise TypeError('invalid integer index set I')
        if len(I) > 0 and min(I) < 0:
            raise IndexError('negative element in integer index set I')
        if len(I) > 0 and max(I) > n - 1:
            raise IndexError('maximum element in in integer index set I is larger than n-1')
        bkc = m * [mosek.boundkey.up] + p * [mosek.boundkey.fx]
        blc = m * [-inf] + [bi for bi in b]
        buc = list(h) + list(b)
        bkx = n * [mosek.boundkey.fr]
        blx = n * [-inf]
        bux = n * [+inf]
        colptr, asub, acof = sparse([G, A]).CCS
        aptrb, aptre = (colptr[:-1], colptr[1:])
        with env.Task(0, 0) as task:
            task.set_Stream(mosek.streamtype.log, streamprinter)
            options = kwargs.get('options', globals()['options'])
            for param, val in options.items():
                if str(param)[:6] == 'iparam':
                    task.putintparam(param, val)
                elif str(param)[:6] == 'dparam':
                    task.putdouparam(param, val)
                elif str(param)[:6] == 'sparam':
                    task.putstrparam(param, val)
                else:
                    raise ValueError('invalid MOSEK parameter: ' + str(param))
            task.inputdata(m + p, n, list(c), 0.0, list(aptrb), list(aptre), list(asub), list(acof), bkc, blc, buc, bkx, blx, bux)
            task.putobjsense(mosek.objsense.minimize)
            if len(I) > 0:
                task.putvartypelist(list(I), len(I) * [mosek.variabletype.type_int])
            task.putintparam(mosek.iparam.mio_mode, mosek.miomode.satisfied)
            if taskfile:
                task.writetask(taskfile)
            task.optimize()
            task.solutionsummary(mosek.streamtype.msg)
            if len(I) > 0:
                solsta = task.getsolsta(mosek.soltype.itg)
            else:
                solsta = task.getsolsta(mosek.soltype.bas)
            x = n * [0.0]
            if len(I) > 0:
                task.getsolutionslice(mosek.soltype.itg, mosek.solitem.xx, 0, n, x)
            else:
                task.getsolutionslice(mosek.soltype.bas, mosek.solitem.xx, 0, n, x)
            x = matrix(x)
    if solsta is mosek.solsta.unknown:
        return (solsta, None)
    else:
        return (solsta, x)
import sys
import numpy as np
from scipy.linalg import norm, solve, inv, qr, svd, LinAlgError
from numpy import asarray, dot, vdot
import scipy.sparse.linalg
import scipy.sparse
from scipy.linalg import get_blas_funcs
import inspect
from scipy._lib._util import getfullargspec_no_self as _getfullargspec
from ._linesearch import scalar_search_wolfe1, scalar_search_armijo
def asjacobian(J):
    """
    Convert given object to one suitable for use as a Jacobian.
    """
    spsolve = scipy.sparse.linalg.spsolve
    if isinstance(J, Jacobian):
        return J
    elif inspect.isclass(J) and issubclass(J, Jacobian):
        return J()
    elif isinstance(J, np.ndarray):
        if J.ndim > 2:
            raise ValueError('array must have rank <= 2')
        J = np.atleast_2d(np.asarray(J))
        if J.shape[0] != J.shape[1]:
            raise ValueError('array must be square')
        return Jacobian(matvec=lambda v: dot(J, v), rmatvec=lambda v: dot(J.conj().T, v), solve=lambda v, tol=0: solve(J, v), rsolve=lambda v, tol=0: solve(J.conj().T, v), dtype=J.dtype, shape=J.shape)
    elif scipy.sparse.issparse(J):
        if J.shape[0] != J.shape[1]:
            raise ValueError('matrix must be square')
        return Jacobian(matvec=lambda v: J @ v, rmatvec=lambda v: J.conj().T @ v, solve=lambda v, tol=0: spsolve(J, v), rsolve=lambda v, tol=0: spsolve(J.conj().T, v), dtype=J.dtype, shape=J.shape)
    elif hasattr(J, 'shape') and hasattr(J, 'dtype') and hasattr(J, 'solve'):
        return Jacobian(matvec=getattr(J, 'matvec'), rmatvec=getattr(J, 'rmatvec'), solve=J.solve, rsolve=getattr(J, 'rsolve'), update=getattr(J, 'update'), setup=getattr(J, 'setup'), dtype=J.dtype, shape=J.shape)
    elif callable(J):

        class Jac(Jacobian):

            def update(self, x, F):
                self.x = x

            def solve(self, v, tol=0):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return solve(m, v)
                elif scipy.sparse.issparse(m):
                    return spsolve(m, v)
                else:
                    raise ValueError('Unknown matrix type')

            def matvec(self, v):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return dot(m, v)
                elif scipy.sparse.issparse(m):
                    return m @ v
                else:
                    raise ValueError('Unknown matrix type')

            def rsolve(self, v, tol=0):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return solve(m.conj().T, v)
                elif scipy.sparse.issparse(m):
                    return spsolve(m.conj().T, v)
                else:
                    raise ValueError('Unknown matrix type')

            def rmatvec(self, v):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return dot(m.conj().T, v)
                elif scipy.sparse.issparse(m):
                    return m.conj().T @ v
                else:
                    raise ValueError('Unknown matrix type')
        return Jac()
    elif isinstance(J, str):
        return dict(broyden1=BroydenFirst, broyden2=BroydenSecond, anderson=Anderson, diagbroyden=DiagBroyden, linearmixing=LinearMixing, excitingmixing=ExcitingMixing, krylov=KrylovJacobian)[J]()
    else:
        raise TypeError('Cannot convert object to a Jacobian')
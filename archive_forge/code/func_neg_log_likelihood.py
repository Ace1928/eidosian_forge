import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular, cho_factor, cho_solve
from ase.optimize.gpmin.kernel import SquaredExponential
from ase.optimize.gpmin.prior import ZeroPrior
def neg_log_likelihood(self, params, *args):
    """Negative logarithm of the marginal likelihood and its derivative.
        It has been built in the form that suits the best its optimization,
        with the scipy minimize module, to find the optimal hyperparameters.

        Parameters:

        l: The scale for which we compute the marginal likelihood
        *args: Should be a tuple containing the inputs and targets
               in the training set-
        """
    X, Y = args
    self.kernel.set_params(np.array([params[0], params[1], self.noise]))
    self.train(X, Y)
    y = Y.flatten()
    logP = -0.5 * np.dot(y - self.m, self.a) - np.sum(np.log(np.diag(self.L))) - X.shape[0] * 0.5 * np.log(2 * np.pi)
    grad = self.kernel.gradient(X)
    D_P_input = np.array([np.dot(np.outer(self.a, self.a), g) for g in grad])
    D_complexity = np.array([cho_solve((self.L, self.lower), g) for g in grad])
    DlogP = 0.5 * np.trace(D_P_input - D_complexity, axis1=1, axis2=2)
    return (-logP, -DlogP)
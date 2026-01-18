import numpy as np
from scipy.special import expit as logistic_sigmoid
from scipy.special import xlogy
def inplace_identity_derivative(Z, delta):
    """Apply the derivative of the identity function: do nothing.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the identity activation function during
        the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
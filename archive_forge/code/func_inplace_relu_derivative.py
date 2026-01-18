import numpy as np
from scipy.special import expit as logistic_sigmoid
from scipy.special import xlogy
def inplace_relu_derivative(Z, delta):
    """Apply the derivative of the relu function.

    It exploits the fact that the derivative is a simple function of the output
    value from rectified linear units activation function.

    Parameters
    ----------
    Z : {array-like, sparse matrix}, shape (n_samples, n_features)
        The data which was output from the rectified linear units activation
        function during the forward pass.

    delta : {array-like}, shape (n_samples, n_features)
         The backpropagated error signal to be modified inplace.
    """
    delta[Z == 0] = 0
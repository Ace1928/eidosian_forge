import numpy as np
def linear_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray) -> tuple:
    """
    Implement the linear part of a layer's forward propagation.

    Args:
        A (np.ndarray): Activations from previous layer (or input data).
        W (np.ndarray): Weights matrix.
        b (np.ndarray): Bias vector.

    Returns:
        tuple: The linear cache and the linear hypothesis Z.
    """
    Z = W.dot(A) + b
    cache = (A, W, b)
    return (Z, cache)
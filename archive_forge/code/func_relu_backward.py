import numpy as np
def relu_backward(dA: np.ndarray, cache: np.ndarray) -> np.ndarray:
    """
    Implements the backward propagation for a single ReLU unit.

    Args:
        dA (np.ndarray): Post-activation gradient.
        cache (np.ndarray): 'Z' where we store for computing backward propagation efficiently.

    Returns:
        np.ndarray: Gradient of the cost with respect to Z.
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ
import numpy as np
def linear_activation_backward(dA: np.ndarray, cache: tuple, activation: str, lambd: float) -> tuple:
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Args:
        dA (np.ndarray): Post-activation gradient for current layer l.
        cache (tuple): Tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently.
        activation (str): The activation to be used in this layer, stored as a text string: "sigmoid" or "relu".
        lambd (float): Regularization hyperparameter.

    Returns:
        tuple: Gradients of the cost with respect to A_prev, W, and b respectively.
    """
    linear_cache, activation_cache = cache
    if activation == 'relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache, lambd)
    return (dA_prev, dW, db)
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
import sklearn.model_selection
def backward_propagation(AL: np.ndarray, Y: np.ndarray, caches: List[Tuple[np.ndarray]], lambd: float) -> Dict[str, np.ndarray]:
    """
    Implements the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    """
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = linear_activation_backward(dAL, current_cache, activation='sigmoid', lambd=lambd)
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 2)], current_cache, activation='relu', lambd=lambd)
        grads['dA' + str(l + 1)] = dA_prev_temp
        grads['dW' + str(l + 1)] = dW_temp
        grads['db' + str(l + 1)] = db_temp
    return grads
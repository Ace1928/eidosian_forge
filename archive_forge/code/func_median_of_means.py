import warnings
from collections.abc import Iterable
from string import ascii_letters as ABC
import numpy as np
import pennylane as qml
def median_of_means(arr, num_batches, axis=0):
    """
    The median of means of the given array.

    The array is split into the specified number of batches. The mean value
    of each batch is taken, then the median of the mean values is returned.

    Args:
        arr (tensor-like[float]): The 1-D array for which the median of means
            is determined
        num_batches (int): The number of batches to split the array into

    Returns:
        float: The median of means
    """
    batch_size = int(np.ceil(arr.shape[0] / num_batches))
    means = [qml.math.mean(arr[i * batch_size:(i + 1) * batch_size], 0) for i in range(num_batches)]
    return np.median(means, axis=axis)
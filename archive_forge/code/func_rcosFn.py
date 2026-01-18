import numpy as np
import scipy.misc as sc
import scipy.signal
import scipy.io
def rcosFn(self, width, position):
    N = 256
    X = np.pi * np.array(range(-N - 1, 2))
    X /= 2.0 * N
    Y = np.cos(X) ** 2
    Y[0] = Y[1]
    Y[N + 2] = Y[N + 1]
    X = position + 2 * width / np.pi * (X + np.pi / 4)
    return (X, Y)
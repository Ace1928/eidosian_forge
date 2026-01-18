import math
def cost_svd(x):
    A, = x.deps
    shape = A.shape
    m = max(shape)
    n = min(shape)
    return 4 * m * n ** 2 - 4 / 3 * n ** 3
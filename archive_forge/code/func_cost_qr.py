import math
def cost_qr(x):
    A, = x.deps
    shape = A.shape
    m = max(shape)
    n = min(shape)
    return 2 * m * n ** 2 - 2 / 3 * n ** 3
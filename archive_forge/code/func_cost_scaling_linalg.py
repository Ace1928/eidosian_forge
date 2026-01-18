import math
def cost_scaling_linalg(x):
    """Here we only care about the leading factor of the cost, which we need to
    preserve so that we can prime number decompose it.
    """
    A, = x.deps
    shape = A.shape
    m = max(shape)
    n = min(shape)
    return m * n ** 2
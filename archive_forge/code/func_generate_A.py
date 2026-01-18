import os
def generate_A(K):
    A = [Poly(1, x)]
    for k in range(K):
        A.append(Poly(1 - 2 * k * x, x) * A[k] + Poly(x * (x + 1)) * A[k].diff())
    return A
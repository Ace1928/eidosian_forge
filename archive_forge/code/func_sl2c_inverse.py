from ..matrix import matrix
def sl2c_inverse(A):
    return matrix([[A[1, 1], -A[0, 1]], [-A[1, 0], A[0, 0]]])
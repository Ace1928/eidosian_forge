import re
import string
def as_two_by_two_matrices(L):
    assert len(L) % 4 == 0
    return [[(L[i], L[i + 1]), (L[i + 2], L[i + 3])] for i in range(0, len(L), 4)]
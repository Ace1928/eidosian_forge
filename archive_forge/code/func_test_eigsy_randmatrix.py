from mpmath import mp
from mpmath import libmp
def test_eigsy_randmatrix():
    N = 5
    for a in xrange(10):
        A = 2 * mp.randmatrix(N, N) - 1
        for i in xrange(0, N):
            for j in xrange(i + 1, N):
                A[j, i] = A[i, j]
        run_eigsy(A)
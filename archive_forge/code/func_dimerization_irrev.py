from .._util import get_backend
def dimerization_irrev(t, kf, initial_C, P0=1, t0=0):
    return 1 / (1 / initial_C + 2 * kf * (t - t0))
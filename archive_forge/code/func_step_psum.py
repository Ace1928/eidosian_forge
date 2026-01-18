from ..libmp.backend import xrange
from .calculus import defun
def step_psum(self, s):
    """
        This routine applies the convergence acceleration to the partial sums.

        A   = sum(a_k, k = 0..infinity)
        s_n = sum(a_k, k = 0..n)

        v, e = ...step_psum(s_k)

        output:
          v      current estimate of the series A
          e      an error estimate which is simply the difference between the current
                 estimate and the last estimate.
        """
    if self.variant != 'v':
        if self.n == 0:
            self.last_s = s
            self.run(s, s)
        else:
            self.run(s, s - self.last_s)
            self.last_s = s
    else:
        if isinstance(self.last_s, bool):
            self.last_s = s
            self.last_w = s
            self.last = 0
            return (s, abs(s))
        na1 = s - self.last_s
        self.run(self.last_s, self.last_w, na1)
        self.last_w = na1
        self.last_s = s
    value = self.A[0] / self.B[0]
    err = abs(value - self.last)
    self.last = value
    return (value, err)
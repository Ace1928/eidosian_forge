import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial
def add_xi(self, xi, yi=None):
    """Add more x values to the set to be interpolated.

        The barycentric interpolation algorithm allows easy updating
        by adding more points for the polynomial to pass through.

        Parameters
        ----------
        xi : cupy.ndarray
            The x-coordinates of the points that the polynomial should
            pass through
        yi : cupy.ndarray, optional
            The y-coordinates of the points the polynomial should pass
            through. Should have shape ``(xi.size, R)``; if R > 1 then
            the polynomial is vector-valued
            If `yi` is not given, the y values will be supplied later.
            `yi` should be given if and only if the interpolator has y
            values specified

        """
    if yi is not None:
        if self.yi is None:
            raise ValueError('No previous yi value to update!')
        yi = self._reshape_yi(yi, check=True)
        self.yi = cupy.vstack((self.yi, yi))
    elif self.yi is not None:
        raise ValueError('No update to yi provided!')
    old_n = self.n
    self.xi = cupy.concatenate((self.xi, xi))
    self.n = len(self.xi)
    self.wi **= -1
    old_wi = self.wi
    self.wi = cupy.zeros(self.n)
    self.wi[:old_n] = old_wi
    for j in range(old_n, self.n):
        self.wi[:j] *= self._inv_capacity * (self.xi[j] - self.xi[:j])
        self.wi[j] = cupy.prod(self._inv_capacity * (self.xi[:j] - self.xi[j]))
    self.wi **= -1
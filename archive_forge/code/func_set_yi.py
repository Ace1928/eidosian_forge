import cupy
from cupyx.scipy._lib._util import _asarray_validated, float_factorial
def set_yi(self, yi, axis=None):
    """Update the y values to be interpolated.

        The barycentric interpolation algorithm requires the calculation
        of weights, but these depend only on the xi. The yi can be changed
        at any time.

        Parameters
        ----------
        yi : cupy.ndarray
            The y-coordinates of the points the polynomial should pass
            through. If None, the y values will be supplied later.
        axis : int, optional
            Axis in the yi array corresponding to the x-coordinate values

        """
    if yi is None:
        self.yi = None
        return
    self._set_yi(yi, xi=self.xi, axis=axis)
    self.yi = self._reshape_yi(yi)
    self.n, self.r = self.yi.shape
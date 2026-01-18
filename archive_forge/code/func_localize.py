import numpy as np
from pygsp import utils
from . import approximations
def localize(self, i, **kwargs):
    """Localize the kernels at a node (to visualize them).

        That is particularly useful to visualize a filter in the vertex domain.

        A kernel is localized by filtering a Kronecker delta, i.e.

        .. math:: g(L) s = g(L)_i, \\text{ where } s_j = \\delta_{ij} =
                  \\begin{cases} 0 \\text{ if } i \\neq j \\\\
                                1 \\text{ if } i = j    \\end{cases}

        Parameters
        ----------
        i : int
            Index of the node where to localize the kernel.
        kwargs: dict
            Parameters to be passed to the :meth:`analyze` method.

        Returns
        -------
        s : ndarray
            Kernel localized at vertex i.

        Examples
        --------
        Visualize heat diffusion on a grid by localizing the heat kernel.

        >>> import matplotlib
        >>> N = 20
        >>> DELTA = N//2 * (N+1)
        >>> G = graphs.Grid2d(N)
        >>> G.estimate_lmax()
        >>> g = filters.Heat(G, 100)
        >>> s = g.localize(DELTA)
        >>> G.plot_signal(s, highlight=DELTA)

        """
    s = np.zeros(self.G.N)
    s[i] = 1
    return np.sqrt(self.G.N) * self.filter(s, **kwargs)
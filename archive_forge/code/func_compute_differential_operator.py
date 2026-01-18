import numpy as np
from scipy import sparse
from pygsp import utils
def compute_differential_operator(self):
    """Compute the graph differential operator (cached).

        The differential operator is a matrix such that

        .. math:: L = D^T D,

        where :math:`D` is the differential operator and :math:`L` is the graph
        Laplacian. It is used to compute the gradient and the divergence of a
        graph signal, see :meth:`grad` and :meth:`div`.

        The result is cached and accessible by the :attr:`D` property.

        See also
        --------
        grad : compute the gradient
        div : compute the divergence

        Examples
        --------
        >>> G = graphs.Logo()
        >>> G.N, G.Ne
        (1130, 3131)
        >>> G.compute_differential_operator()
        >>> G.D.shape == (G.Ne, G.N)
        True

        """
    v_in, v_out, weights = self.get_edge_list()
    n = len(v_in)
    Dr = np.concatenate((np.arange(n), np.arange(n)))
    Dc = np.empty(2 * n)
    Dc[:n] = v_in
    Dc[n:] = v_out
    Dv = np.empty(2 * n)
    if self.lap_type == 'combinatorial':
        Dv[:n] = np.sqrt(weights)
        Dv[n:] = -Dv[:n]
    elif self.lap_type == 'normalized':
        Dv[:n] = np.sqrt(weights / self.dw[v_in])
        Dv[n:] = -np.sqrt(weights / self.dw[v_out])
    else:
        raise ValueError('Unknown lap_type {}'.format(self.lap_type))
    self._D = sparse.csc_matrix((Dv, (Dr, Dc)), shape=(n, self.N))
import logging
import itertools
import os
import heapq
import warnings
import numpy
import scipy.sparse
from gensim import interfaces, utils, matutils
def get_document_id(self, pos):
    """Get index vector at position `pos`.

        Parameters
        ----------
        pos : int
            Vector position.

        Return
        ------
        {:class:`scipy.sparse.csr_matrix`, :class:`numpy.ndarray`}
            Index vector. Type depends on underlying index.

        Notes
        -----
        The vector is of the same type as the underlying index (ie., dense for
        :class:`~gensim.similarities.docsim.MatrixSimilarity`
        and scipy.sparse for :class:`~gensim.similarities.docsim.SparseMatrixSimilarity`.

        """
    assert 0 <= pos < len(self), 'requested position out of range'
    return self.get_index().index[pos]
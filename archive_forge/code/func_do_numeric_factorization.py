from scipy.sparse import isspmatrix_coo, coo_matrix, tril, spmatrix
import numpy as np
from .base import DirectLinearSolverInterface, LinearSolverResults, LinearSolverStatus
from typing import Union, Tuple, Optional
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
def do_numeric_factorization(self, matrix: Union[spmatrix, BlockMatrix], raise_on_error: bool=True) -> LinearSolverResults:
    """
        Perform Mumps factorization. Note that do_symbolic_factorization should be called
        before do_numeric_factorization.

        Parameters
        ----------
        matrix: scipy.sparse.spmatrix or pyomo.contrib.pynumero.sparse.BlockMatrix
            This matrix must have the same nonzero structure as the matrix passed into
            do_symbolic_factorization. The matrix will be converted to coo format if it
            is not already in coo format. If sym is 1 or 2, the matrix will be converted
            to lower triangular.
        """
    if self._nnz is None:
        raise RuntimeError('Call do_symbolic_factorization first.')
    if not isspmatrix_coo(matrix):
        matrix = matrix.tocoo()
    if self._sym in {1, 2}:
        matrix = tril(matrix)
    nrows, ncols = matrix.shape
    if nrows != ncols:
        raise ValueError('matrix is not square')
    if self._dim != nrows:
        raise ValueError('The shape of the matrix changed between symbolic and numeric factorization')
    if self._nnz != matrix.nnz:
        raise ValueError('The number of nonzeros changed between symbolic and numeric factorization')
    try:
        self._mumps.set_centralized_assembled_values(matrix.data)
        self._mumps.run(job=2)
    except RuntimeError as err:
        if raise_on_error:
            raise err
    stat = self.get_infog(1)
    res = LinearSolverResults()
    if stat == 0:
        res.status = LinearSolverStatus.successful
    elif stat in {-6, -10}:
        res.status = LinearSolverStatus.singular
    elif stat in {-8, -9, -19}:
        res.status = LinearSolverStatus.not_enough_memory
    elif stat < 0:
        res.status = LinearSolverStatus.error
    else:
        res.status = LinearSolverStatus.warning
    return res
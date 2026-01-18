from scipy.sparse import isspmatrix_coo, coo_matrix, tril, spmatrix
import numpy as np
from .base import DirectLinearSolverInterface, LinearSolverResults, LinearSolverStatus
from typing import Union, Tuple, Optional
from pyomo.common.dependencies import attempt_import
from pyomo.contrib.pynumero.sparse import BlockVector, BlockMatrix
def do_back_solve(self, rhs: Union[np.ndarray, BlockVector], raise_on_error: bool=True) -> Tuple[Optional[Union[np.ndarray, BlockVector]], LinearSolverResults]:
    """
        Perform back solve with Mumps. Note that both do_symbolic_factorization and
        do_numeric_factorization should be called before do_back_solve.

        Parameters
        ----------
        rhs: numpy.ndarray or pyomo.contrib.pynumero.sparse.BlockVector
            The right hand side in matrix * x = rhs.

        Returns
        -------
        result: numpy.ndarray or pyomo.contrib.pynumero.sparse.BlockVector
            The x in matrix * x = rhs. If rhs is a BlockVector, then, result
            will be a BlockVector with the same block structure as rhs.
        """
    if isinstance(rhs, BlockVector):
        _rhs = rhs.flatten()
        result = _rhs
    else:
        result = rhs.copy()
    self._mumps.set_rhs(result)
    self._mumps.run(job=3)
    if isinstance(rhs, BlockVector):
        _result = rhs.copy_structure()
        _result.copyfrom(result)
        result = _result
    return (result, LinearSolverResults(LinearSolverStatus.successful))
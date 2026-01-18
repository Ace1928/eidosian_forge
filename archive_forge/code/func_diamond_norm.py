from __future__ import annotations
import logging
import numpy as np
from qiskit.exceptions import QiskitError, MissingOptionalLibraryError
from qiskit.circuit.gate import Gate
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel import Choi, SuperOp
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.states.measures import state_fidelity
from qiskit.utils import optionals as _optionals
def diamond_norm(choi: Choi | QuantumChannel, solver: str='SCS', **kwargs) -> float:
    """Return the diamond norm of the input quantum channel object.

    This function computes the completely-bounded trace-norm (often
    referred to as the diamond-norm) of the input quantum channel object
    using the semidefinite-program from reference [1].

    Args:
        choi(Choi or QuantumChannel): a quantum channel object or
                                      Choi-matrix array.
        solver (str): The solver to use.
        kwargs: optional arguments to pass to CVXPY solver.

    Returns:
        float: The completely-bounded trace norm :math:`\\|\\mathcal{E}\\|_{\\diamond}`.

    Raises:
        QiskitError: if CVXPY package cannot be found.

    Additional Information:
        The input to this function is typically *not* a CPTP quantum
        channel, but rather the *difference* between two quantum channels
        :math:`\\|\\Delta\\mathcal{E}\\|_\\diamond` where
        :math:`\\Delta\\mathcal{E} = \\mathcal{E}_1 - \\mathcal{E}_2`.

    Reference:
        J. Watrous. "Simpler semidefinite programs for completely bounded
        norms", arXiv:1207.5726 [quant-ph] (2012).

    .. note::

        This function requires the optional CVXPY package to be installed.
        Any additional kwargs will be passed to the ``cvxpy.solve``
        function. See the CVXPY documentation for information on available
        SDP solvers.
    """
    from scipy import sparse
    cvxpy = _cvxpy_check('`diamond_norm`')
    choi = Choi(_input_formatter(choi, Choi, 'diamond_norm', 'choi'))

    def cvx_bmat(mat_r, mat_i):
        """Block matrix for embedding complex matrix in reals"""
        return cvxpy.bmat([[mat_r, -mat_i], [mat_i, mat_r]])
    dim_in = choi._input_dim
    dim_out = choi._output_dim
    size = dim_in * dim_out
    r0_r = cvxpy.Variable((dim_in, dim_in))
    r0_i = cvxpy.Variable((dim_in, dim_in))
    r0 = cvx_bmat(r0_r, r0_i)
    r1_r = cvxpy.Variable((dim_in, dim_in))
    r1_i = cvxpy.Variable((dim_in, dim_in))
    r1 = cvx_bmat(r1_r, r1_i)
    x_r = cvxpy.Variable((size, size))
    x_i = cvxpy.Variable((size, size))
    iden = sparse.eye(dim_out)
    c_r = cvxpy.bmat([[cvxpy.kron(iden, r0_r), x_r], [x_r.T, cvxpy.kron(iden, r1_r)]])
    c_i = cvxpy.bmat([[cvxpy.kron(iden, r0_i), x_i], [-x_i.T, cvxpy.kron(iden, r1_i)]])
    c = cvx_bmat(c_r, c_i)
    choi_rt = np.transpose(np.reshape(choi.data, (dim_in, dim_out, dim_in, dim_out)), (3, 2, 1, 0)).reshape(choi.data.shape)
    choi_rt_r = choi_rt.real
    choi_rt_i = choi_rt.imag
    cons = [r0 >> 0, r0_r == r0_r.T, r0_i == -r0_i.T, cvxpy.trace(r0_r) == 1, r1 >> 0, r1_r == r1_r.T, r1_i == -r1_i.T, cvxpy.trace(r1_r) == 1, c >> 0]
    obj = cvxpy.Maximize(cvxpy.trace(choi_rt_r @ x_r) + cvxpy.trace(choi_rt_i @ x_i))
    prob = cvxpy.Problem(obj, cons)
    sol = prob.solve(solver=solver, **kwargs)
    return sol